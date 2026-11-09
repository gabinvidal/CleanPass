// CleanPass - Nuke node
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// One node, three cleaning stages, each in its own tab:
//   Firefly   - removes outliers only, every other pixel stays bit-for-bit
//   Deflicker - motion-compensated temporal stabilisation
//   Denoise   - Intel Open Image Denoise
// The algorithms live in cp_core / cp_deflicker / cp_pipeline (no Nuke
// dependency); this file only moves pixels in and out of Nuke.

#include <DDImage/Channel.h>
#include <DDImage/Hash.h>
#include <DDImage/ImagePlane.h>
#include <DDImage/Knob.h>
#include <DDImage/Knobs.h>
#include <DDImage/PlanarIop.h>
#include <DDImage/RequestOutput.h>
#include <DDImage/Row.h>

#include "cp_lock.h"
#include "cp_oidn.h"
#include "cp_pipeline.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace DD::Image;

namespace
{

const char* const kClass = "CleanPass";
#define CP_VERSION "1.0.0"
const char* const kHelp =
    "<b>CleanPass</b> - CG render cleaning in three stages, each in its own tab:\n\n"
    "<b>Firefly</b>: finds the outlier pixels (against their neighbourhood, an OIDN or statistical "
    "expected value, and the neighbour frames) and corrects only them. Every other pixel is untouched.\n"
    "<b>Deflicker</b>: matches every pixel in the neighbour frames (block matching on the beauty and the "
    "albedo / normal guides) and averages it with the samples of the same surface. Removes boiling and "
    "shimmering without ghosting.\n"
    "<b>Denoise</b>: Intel Open Image Denoise, guided by albedo / normal, stabilised over time.\n\n"
    "Motion vectors, depth and variance passes are used when present (all optional). Light AOVs "
    "(light groups) can follow every stage and still add up to the beauty.\n\n"
    "Gabin Vidal";

const char* const kReferenceNames[] = {"OIDN (machine learning)", "Statistical (fast)", nullptr};
const char* const kMethodNames[] = {"Expected value", "Local median", "Clamp", nullptr};
const char* const kViewNames[] = {"Result", "Fireflies overlay", "Removed", "Firefly matte", "Deflicker weight", nullptr};
const char* const kDeviceNames[] = {"CPU", "CUDA", nullptr};
const char* const kQualityNames[] = {"Balanced", "High", "Fast (preview)", nullptr};
const char* const kFramesNames[] = {"1  (t-1, t+1)", "2  (t-2 ... t+2)", nullptr};

enum View
{
    eResult = 0,
    eOverlay = 1,
    eRemoved = 2,
    eMatte = 3,
    eDfWeight = 4
};

// frame offset of the copy `o` of input 0 (see split_input)
int offsetOfCopy(int o)
{
    static const int kOff[5] = {0, -1, 1, -2, 2};
    return (o >= 0 && o < 5) ? kOff[o] : 0;
}

// Up to three channels of a layer, indexed by colour (r, g, b).
struct Rgb
{
    Channel ch[3] = {Chan_Black, Chan_Black, Chan_Black};
    ChannelSet set;
    std::string name;
    bool valid() const { return ch[0] != Chan_Black || ch[1] != Chan_Black || ch[2] != Chan_Black; }
};

Rgb rgbOf(const ChannelSet& cs)
{
    Rgb r;
    foreach (z, cs)
    {
        const unsigned ci = colourIndex(z);
        if (ci < 3 && r.ch[ci] == Chan_Black)
        {
            r.ch[ci] = z;
            r.set += z;
            if (r.name.empty())
                r.name = getLayerName(z);
        }
    }
    // single-channel layers: use that channel for the three components
    if (r.valid())
        for (int c = 0; c < 3; ++c)
            if (r.ch[c] == Chan_Black)
                for (int d = 0; d < 3; ++d)
                    if (r.ch[d] != Chan_Black)
                    {
                        r.ch[c] = r.ch[d];
                        break;
                    }
    return r;
}

Box bboxOf(Iop* iop)
{
    const Box b = iop->info();
    return b;
}

// Copies up to three channels of `iop` over `box` into a dense planar buffer
// (3 * w * h, rows from box.y()). Reads rows only inside the input's own bbox
// (zero elsewhere) through the row API, which is always safe.
void fetchDense(Iop* iop, const Box& box, const Channel ch[3], std::vector<float>& out)
{
    const int w = box.w(), h = box.h();
    out.assign((size_t)w * h * 3, 0.f);
    if (!iop || w <= 0 || h <= 0)
        return;
    Box inter = box;
    inter.intersect(bboxOf(iop));
    if (inter.w() <= 0 || inter.h() <= 0)
        return;
    ChannelSet cs;
    for (int c = 0; c < 3; ++c)
        if (ch[c] != Chan_Black && iop->info().channels().contains(ch[c]))
            cs += ch[c];
    if (cs.empty())
        return;
    Row row(inter.x(), inter.r());
    for (int y = inter.y(); y < inter.t(); ++y)
    {
        iop->get(y, inter.x(), inter.r(), cs, row);
        for (int c = 0; c < 3; ++c)
        {
            if (ch[c] == Chan_Black || !cs.contains(ch[c]))
                continue;
            const float* src = row[ch[c]];
            if (!src)
                continue;
            float* dst = out.data() + (size_t)c * w * h + (size_t)(y - box.y()) * w;
            for (int x = inter.x(); x < inter.r(); ++x)
                dst[x - box.x()] = src[x];
        }
    }
}

// Usual layer names of the optional passes (case-insensitive). An empty layer
// knob means "find it by name".
enum Pass
{
    eAlbedo,
    eNormal,
    eMotion,
    eDepth,
    eVariance,
    ePassCount
};
const char* const* passNames(int pass)
{
    static const char* const albedo[] = {"albedo", "diffuse_albedo", "denoise_albedo", "albedo_diffuse",
                                         "diffusealbedo", "alb", nullptr};
    static const char* const normal[] = {"N", "normal", "normals", "Nworld", "N_world", "denoise_normal", nullptr};
    static const char* const motion[] = {"motionvector", "motionvectors", "motion_vector", "motion", "mv",
                                         "velocity", "vel", nullptr};
    static const char* const depth[] = {"depth", "Z", "zdepth", "z_depth", "Pz", nullptr};
    static const char* const variance[] = {"RGBA_variance", "beauty_variance", "variance", "rgba_var", nullptr};
    static const char* const* const all[] = {albedo, normal, motion, depth, variance};
    return all[pass];
}

std::string lowerOf(std::string t)
{
    for (char& c : t)
        c = (char)std::tolower((unsigned char)c);
    return t;
}

// Name of the layer of `chans` that matches a pass, or "".
std::string findPassLayer(const ChannelSet& chans, int pass)
{
    std::vector<std::string> layers;
    foreach (z, chans)
    {
        const std::string l = getLayerName(z);
        if (std::find(layers.begin(), layers.end(), l) == layers.end())
            layers.push_back(l);
    }
    for (const char* const* n = passNames(pass); *n; ++n)
        for (const std::string& l : layers)
            if (lowerOf(l) == lowerOf(*n))
                return l;
    if (pass == eVariance)
        for (const std::string& l : layers)
            if (lowerOf(l).find("variance") != std::string::npos)
                return l;
    return std::string();
}

// Shell-style wildcard match (* and ?), case-insensitive.
bool globMatch(const char* p, const char* t)
{
    if (!*p)
        return !*t;
    if (*p == '*')
        return globMatch(p + 1, t) || (*t && globMatch(p, t + 1));
    if (*t && (*p == '?' || std::tolower((unsigned char)*p) == std::tolower((unsigned char)*t)))
        return globMatch(p + 1, t + 1);
    return false;
}

cleanpass::View3 viewOf(const std::vector<float>& dense, const Box& box)
{
    if (dense.empty())
        return cleanpass::View3();
    const size_t n = (size_t)box.w() * box.h();
    return cleanpass::View3::planar(dense.data(), dense.data() + n, dense.data() + 2 * n, box.w());
}

} // namespace

class CleanPassIop : public PlanarIop
{
public:
    explicit CleanPassIop(Node* node);

    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 4; }
    const char* input_label(int n, char*) const override;
    int split_input(int n) const override;
    const OutputContext& inputContext(int n, int offset, OutputContext& ctx) const override;

    void knobs(Knob_Callback f) override;
    int knob_changed(Knob* k) override;

    void _validate(bool for_real) override;
    void getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& req) const override;
    void renderStripe(ImagePlane& plane) override;

    bool useStripes() const override { return false; }
    bool renderFullPlanes() const override { return true; }
    PackedPreference packedPreference() const override { return ePackedPreferenceUnpacked; }

    const char* Class() const override { return d.name; }
    const char* node_help() const override { return kHelp; }
    static const Iop::Description d;

private:
    // ---- knobs: Firefly ----
    bool m_ffOn = true;
    int m_reference = 0;
    float m_threshold, m_softness, m_isolation, m_adaptivity, m_minLevel;
    int m_size, m_grow;
    bool m_fixNonFinite;
    int m_method = 0;
    bool m_conserve;
    float m_energyRadius, m_energyAmount;
    bool m_temporal;
    int m_motionRadius;

    // ---- knobs: Deflicker ----
    bool m_dfOn = false;
    int m_dfFrames = 0; // enum index: 0 -> 1 frame, 1 -> 2 frames
    int m_dfSearch, m_dfPatch;
    float m_dfTolerance, m_dfAmount;
    bool m_dfGuides;
    bool m_dfVectors = true, m_dfDepth = true, m_dfVariance = true;

    // ---- knobs: Denoise ----
    bool m_dnOn = false;
    float m_dnAmount = 1.f;
    int m_quality = 1;
    bool m_prefilterAux = false;
    bool m_stabilize = true;
    float m_stabTol = 0.04f;
    bool m_denoiseAovs = true;
    bool m_aovsMatch = true;

    // ---- knobs: Setup ----
    ChannelSet m_beautyLayer, m_albedoLayer, m_normalLayer;
    ChannelSet m_motionLayer, m_depthLayer, m_varianceLayer;
    bool m_propagate = false;
    ChannelSet m_aovLayer[4];
    const char* m_aovPattern = nullptr;
    int m_view = eResult;
    bool m_outputMatte = false;
    Channel m_matteChannel;
    Channel m_maskChannel = Chan_Alpha;
    bool m_invertMask = false;
    float m_mix = 1.f;
    int m_device = 0, m_maxMem = 0;

    // ---- snapshot of everything a render needs, rebuilt by _validate ----
    // Render threads only ever read an immutable Config through a shared_ptr,
    // so a re-validation on the main thread can never pull data from under them.
    struct Config
    {
        Rgb beauty, albedo, normal;
        Rgb motion, depth, variance; // optional passes of src (invalid = not used)
        std::vector<Rgb> aovs;
        Box box; // processed area: bbox inside the format
        cleanpass::PipelineParams pipe;
        cleanpass::OidnSettings oidn;
        bool albedoInput = false, normalInput = false, maskInput = false;
        Channel maskChannel = Chan_Black;
        bool invertMask = false;
        int view = eResult;
        float mix = 1.f;
        bool outputMatte = false;
        Channel matteChannel = Chan_Black;
        int copies = 1; // copies of input 0 the render may read (1 + 2 * radius)
        bool active() const { return pipe.firefly || pipe.deflicker || pipe.denoise; }
    };
    mutable cleanpass::Lock m_cfgLock;
    std::shared_ptr<const Config> m_cfg;
    std::shared_ptr<const Config> config() const;

    // ---- per-frame result cache ----
    struct Result
    {
        cleanpass::PipelineResult r;
        std::vector<float> mask;      // w*h global mask (empty = none)
        std::vector<int> nbCopy;      // PipelineInput::neighbours[i] -> copy index of input 0
        std::vector<float> albedo, normal; // guides of frame t (for the AOV denoise)
        // every AOV, computed together on the first AOV request (the denoise
        // shares the beauty's corrections between them)
        cleanpass::Lock aovLock;
        bool aovDone = false;
        std::vector<std::vector<float>> aovOut;
    };
    cleanpass::Lock m_lock;
    Hash m_cacheKey;
    std::shared_ptr<Result> m_cache;
    // denoised results of single frames, reused by the stabilisation of the
    // next / previous frame (key: the frames and settings that produced them)
    // (a "full" entry also keeps the firefly / deflicker analysis of that frame:
    // made by the stabilisation of a neighbour, it saves the frame its own
    // firefly, deflicker and denoise stages when its turn comes)
    struct Denoised
    {
        Hash key;
        std::shared_ptr<const cleanpass::PipelineResult> res; // res->beauty: the denoised image
    };
    cleanpass::Lock m_dnLock;
    std::vector<Denoised> m_dnCache;
    std::shared_ptr<const cleanpass::PipelineResult> denoisedGet(const Hash& k);
    void denoisedPut(const Hash& k, std::shared_ptr<const cleanpass::PipelineResult> r, bool replace = false);
    void computeAovs(const Config& cfg, Result& res);
    void autoLayers(bool all);

    // ---- helpers ----
    int temporalRadius() const;
    int actualCopies() const;
    Iop* in(int n, int offset = 0) const;
    bool connected(int n) const;
    Hash analysisKey(const Config& cfg) const;
    Hash settingsKey(const Config& cfg) const; // the settings only (no frame, no input)
    std::shared_ptr<Result> analysis(const Config& cfg);
    std::shared_ptr<Result> computeAnalysis(const Config& cfg);
    void updateKnobStates();
};

// ===========================================================================

CleanPassIop::CleanPassIop(Node* node) : PlanarIop(node)
{
    const cleanpass::Params p;
    m_reference = (int)p.reference;
    m_threshold = p.threshold;
    m_softness = p.softness;
    m_isolation = p.isolation;
    m_adaptivity = p.adaptivity;
    m_minLevel = p.minLevel;
    m_size = p.size;
    m_grow = p.grow;
    m_fixNonFinite = p.fixNonFinite;
    m_method = (int)p.method;
    m_conserve = p.conserveEnergy;
    m_energyRadius = p.energyRadius;
    m_energyAmount = p.energyAmount;
    m_temporal = p.temporal;
    m_motionRadius = p.motionRadius;

    const cleanpass::DeflickerParams dp;
    m_dfFrames = dp.frames >= 2 ? 1 : 0;
    m_dfSearch = dp.search;
    m_dfPatch = dp.patch;
    m_dfTolerance = dp.tolerance;
    m_dfAmount = dp.amount;
    m_dfGuides = dp.useGuides;

    m_beautyLayer = Mask_RGB;
    m_matteChannel = getChannel("firefly.matte");
}

const char* CleanPassIop::input_label(int n, char*) const
{
    switch (n)
    {
    case 0: return "src";
    case 1: return "albedo";
    case 2: return "normal";
    case 3: return "mask";
    default: return nullptr;
    }
}

int CleanPassIop::temporalRadius() const
{
    int r = 0;
    if (m_ffOn && m_temporal)
        r = 1;
    if (m_dfOn)
        r = std::max(r, m_dfFrames >= 1 ? 2 : 1);
    if (m_dnOn && m_stabilize)
        r = 2; // t-1 and t+1 are denoised too, with their own neighbours
    return r;
}

int CleanPassIop::split_input(int n) const
{
    // Input 0 is always split in 5 (t, t-1, t+1, t-2, t+2). Nuke calls this while
    // rebuilding the tree, before a changed knob is stored, so a split that
    // depends on the knobs would lag one change behind. Only the copies the
    // enabled stages need are requested (see getRequests), so unused ones cost
    // nothing.
    return n == 0 ? 5 : 1;
}

const OutputContext& CleanPassIop::inputContext(int n, int offset, OutputContext& ctx) const
{
    ctx = outputContext();
    if (n == 0 && offset > 0)
        ctx.setFrame(outputContext().frame() + offsetOfCopy(offset));
    return ctx;
}

int CleanPassIop::actualCopies() const
{
    // copies of input 0 that exist (always 5 once the tree is built, see split_input)
    return std::max(1, std::min(5, inputs() - (node_inputs() - 1)));
}

Iop* CleanPassIop::in(int n, int offset) const
{
    if (n < 0 || n >= node_inputs())
        return nullptr;
    if (n == 0 && offset >= actualCopies())
        return nullptr;
    if (n != 0 && offset != 0)
        return nullptr;
    const int idx = inputNumber(n, offset);
    if (idx < 0 || idx >= inputs())
        return nullptr;
    return dynamic_cast<Iop*>(input(idx));
}

bool CleanPassIop::connected(int n) const
{
    return n < node_inputs() && node_input(n, Op::OUTPUT_OP) != nullptr;
}

// ===========================================================================
// knobs
// ===========================================================================

void CleanPassIop::knobs(Knob_Callback f)
{
    // ------------------------------------------------------------ Firefly
    Tab_knob(f, "Firefly");
    Bool_knob(f, &m_ffOn, "firefly_enable", "enable");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Find the outlier pixels and correct only them. Every other pixel stays untouched.");
    Text_knob(f, "<font color=#888>removes fireflies, keeps the grain</font>");
    ClearFlags(f, Knob::STARTLINE);

    Divider(f, "Detection");
    Enumeration_knob(f, &m_reference, kReferenceNames, "reference", "expected value");
    Tooltip(f, "What a pixel is compared to.\n"
               "OIDN: an Open Image Denoise prediction computed on a pre-cleaned copy of the frame, "
               "guided by albedo / normal. Most accurate, slower.\n"
               "Statistical: a guide-aware robust estimate from the neighbouring samples. Very fast.");
    Float_knob(f, &m_threshold, "threshold", "threshold");
    SetRange(f, 0.0, 6.0);
    Tooltip(f, "In stops: how far above its expected value a pixel must be to count as a firefly. "
               "Lower = more aggressive.");
    Float_knob(f, &m_isolation, "isolation", "isolation");
    SetRange(f, 0.0, 4.0);
    Tooltip(f, "In stops: how far the pixel must stand above the bright part of its neighbourhood. "
               "Protects lines, edges and textures. Lower = more aggressive.");
    Float_knob(f, &m_softness, "softness", "softness");
    SetRange(f, 0.0, 3.0);
    Tooltip(f, "In stops: width of the soft transition of the detection.");
    Float_knob(f, &m_adaptivity, "noise_adaptive", "noise adaptive");
    SetRange(f, 0.0, 2.0);
    Tooltip(f, "Raises the threshold where the render is already noisy, so ordinary noise is left alone.");
    Float_knob(f, &m_minLevel, "min_level", "min brightness");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Pixels darker than this (max of r, g, b) are never touched.");
    Int_knob(f, &m_size, "max_size", "max size");
    SetRange(f, 1, 3);
    Tooltip(f, "Largest firefly radius in pixels. Fireflies are tested at every scale up to this one, "
               "which catches the blobs created by wide pixel filters.");
    Int_knob(f, &m_grow, "halo", "halo");
    SetRange(f, 0, 3);
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Radius of the pixel-filter halo also corrected around each firefly "
               "(only where it is brighter than expected).");
    Bool_knob(f, &m_fixNonFinite, "fix_nan", "fix NaN / inf");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Treat NaN and infinite pixels as fireflies.");

    Divider(f, "Temporal check");
    Bool_knob(f, &m_temporal, "temporal", "temporal check");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Compare with the previous and next frames: a firefly never survives from one frame to the "
               "next, a real highlight does. This is what protects tiny real highlights (rivets, glints...). "
               "Ignored on still images.");
    Int_knob(f, &m_motionRadius, "motion_radius", "motion tolerance");
    SetRange(f, 0, 8);
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Search radius in pixels in the neighbour frames. Must cover how far things move "
               "between two frames.");

    Divider(f, "Correction");
    Enumeration_knob(f, &m_method, kMethodNames, "method", "replace with");
    Tooltip(f, "Expected value: the OIDN prediction, or a real neighbouring sample of median brightness "
               "taken on the same surface (keeps the grain).\n"
               "Local median: per-channel median of the neighbours.\n"
               "Clamp: keep the pixel, only bring its brightness down to the threshold.");
    Bool_knob(f, &m_conserve, "conserve_energy", "keep energy");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Where fireflies are dense, they are not accidents but the noise of the light itself (indirect "
               "light, caustics or volumes at low sample counts): deleting them would darken the image. Their "
               "energy is given back as a smooth field (on the albedo when it is given), so the overall "
               "brightness stays that of the render. Isolated fireflies are removed for good.\n"
               "Off: every detected firefly is deleted (cleaner shadows, but darker than the render).");
    Float_knob(f, &m_energyRadius, "energy_radius", "radius");
    SetRange(f, 4.0, 64.0);
    Tooltip(f, "Scale of the smooth field that gives the energy back (pixels).");
    Float_knob(f, &m_energyAmount, "energy_amount", "amount");
    SetRange(f, 0.0, 1.0);

    // ------------------------------------------------------------ Deflicker
    Tab_knob(f, "Deflicker");
    Bool_knob(f, &m_dfOn, "deflicker_enable", "enable");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Stabilise the render over time: boiling GI, shimmering speculars, crawling sub-pixel "
               "detail. Each pixel is matched in the neighbour frames and averaged with the samples of the "
               "same surface. Where the match is poor (occlusion, real change) nothing is mixed: no ghosting.\n\n"
               "With Denoise and 'stabilise over time' on, leave it off: the stabilisation removes the same "
               "flicker without costing detail, and averaging before OIDN can leave soft streaks (fog, glass). "
               "Use it on its own (no denoise), or for hair and fine detail that crawls.");
    Text_knob(f, "<font color=#888>motion-compensated temporal stabilisation</font>");
    ClearFlags(f, Knob::STARTLINE);
    Divider(f, "");
    Enumeration_knob(f, &m_dfFrames, kFramesNames, "df_frames", "frames");
    Tooltip(f, "How many neighbour frames are averaged on each side. 2 is stronger and slower.");
    Float_knob(f, &m_dfAmount, "df_amount", "amount");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Mix of the stabilised result.");
    Float_knob(f, &m_dfTolerance, "df_tolerance", "tolerance");
    SetRange(f, 0.1, 8.0);
    Tooltip(f, "How different a matched sample may be and still be averaged, relative to the noise "
               "measured on the frame. Higher = stronger, lower = safer on fast changes.");
    Int_knob(f, &m_dfSearch, "df_search", "motion search");
    SetRange(f, 1, 8);
    Tooltip(f, "Largest motion between two frames, in pixels. Raise it for fast moves (slower).");
    Int_knob(f, &m_dfPatch, "df_patch", "patch size");
    SetRange(f, 1, 4);
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Radius of the patches compared to find the motion.");
    Bool_knob(f, &m_dfGuides, "df_guides", "use albedo / normal for the motion");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "The guides are noise free: they make the matching much more reliable. They are read from "
               "the src layers set in the Setup tab (not from the separate albedo / normal inputs).");
    Divider(f, "Optional passes (set in the Setup tab)");
    Bool_knob(f, &m_dfVectors, "df_vectors", "motion vectors");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Use the renderer's motion vectors. Their convention (pixels or NDC, forward or backward, "
               "y up or down) is found automatically by checking them against the image. Each pixel still "
               "keeps the image's own match where the vectors do not fit (reflections, refraction, "
               "transparency). Without vectors the motion is estimated from the image.");
    Bool_knob(f, &m_dfDepth, "df_depth", "depth");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Use the depth pass to reject samples that come from another object (disocclusions).");
    Bool_knob(f, &m_dfVariance, "df_variance", "variance");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "Use a variance pass (e.g. Arnold's AOV variance) for an exact per-pixel noise level, "
               "instead of an estimate.");

    // ------------------------------------------------------------ Denoise
    Tab_knob(f, "Denoise");
    Bool_knob(f, &m_dnOn, "denoise_enable", "enable");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Intel Open Image Denoise on the (firefly-cleaned, stabilised) beauty, guided by albedo / "
               "normal. Acts on the beauty only, not on the AOVs.");
    Text_knob(f, "<font color=#888>Intel Open Image Denoise</font>");
    ClearFlags(f, Knob::STARTLINE);
    Divider(f, "");
    Float_knob(f, &m_dnAmount, "denoise_amount", "amount");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Mix of the denoised result. Below 1 keeps some of the original grain.");
    Enumeration_knob(f, &m_quality, kQualityNames, "quality", "quality");
    Tooltip(f, "OIDN quality, also used by the Firefly OIDN expected value.");
    Bool_knob(f, &m_prefilterAux, "prefilter_aux", "prefilter guides");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Let OIDN denoise the albedo / normal guides first. Use it when the guides are noisy.");
    Divider(f, "Temporal");
    Bool_knob(f, &m_stabilize, "stabilize", "stabilise over time");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "A per-frame denoiser leaves soft blotches that change from frame to frame (boiling). "
               "Frames t-1 and t+1 are denoised too and blended in where they agree, following the motion. "
               "About 3x the denoise time on the first frame; when playing or rendering in order, each frame "
               "then reuses what the previous one prepared for it, for about 1.2x the cost of no stabilisation.");
    Float_knob(f, &m_stabTol, "stabilize_tolerance", "tolerance");
    SetRange(f, 0.005, 0.2);
    Tooltip(f, "How different the neighbouring denoised frames may be and still be blended in. "
               "Higher = steadier, lower = safer on fast changes.");
    Divider(f, "AOVs");
    Bool_knob(f, &m_denoiseAovs, "denoise_aovs", "denoise the AOVs");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Also denoise the AOVs set in the Setup tab (propagate to AOVs), with the same guides.");
    Bool_knob(f, &m_aovsMatch, "aovs_match_beauty", "keep their sum = beauty");
    ClearFlags(f, Knob::STARTLINE);
    Tooltip(f, "When the AOVs add up to the beauty (a complete set of light groups, checked on the image), "
               "share the difference so that they add up exactly to the cleaned beauty again.");
    Text_knob(f, "\nTip: OIDN expects uncorrelated pixel noise. Renders made with a wide pixel\n"
                 "filter denoise less well than with a box / 1 px filter.");

    // ------------------------------------------------------------ Setup
    Tab_knob(f, "Setup");
    Divider(f, "Layers");
    Input_ChannelSet_knob(f, &m_beautyLayer, 0, "beauty_layer", "beauty");
    Tooltip(f, "Layer to clean. It is corrected in place; every other layer passes through.");
    Input_ChannelSet_knob(f, &m_albedoLayer, 0, "albedo_layer", "albedo");
    Tooltip(f, "Albedo / diffuse colour pass from the src input. Ignored when the albedo input is connected. "
               "Used by all three stages.");
    Input_ChannelSet_knob(f, &m_normalLayer, 0, "normal_layer", "normal");
    Tooltip(f, "Normal pass from the src input. Ignored when the normal input is connected.");
    Input_ChannelSet_knob(f, &m_motionLayer, 0, "motion_layer", "motion vectors");
    Tooltip(f, "Optional: renderer motion vectors (x in red, y in green), e.g. Arnold's motionvector. "
               "Any common convention works.");
    Input_ChannelSet_knob(f, &m_depthLayer, 0, "depth_layer", "depth");
    Tooltip(f, "Optional: depth pass (Z).");
    Input_ChannelSet_knob(f, &m_varianceLayer, 0, "variance_layer", "variance");
    Tooltip(f, "Optional: variance of the beauty.");
    Button(f, "detect_layers", "detect layers");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Fill the layers above from the usual names (albedo, N, motionvector, depth / Z, variance...). "
               "An empty layer is found by these names anyway; set a layer by hand to use another one.");

    Divider(f, "AOVs");
    Bool_knob(f, &m_propagate, "propagate_aovs", "propagate to AOVs");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Apply the corrections to the light AOVs below, so that they still add up to the corrected "
               "beauty. Pick additive light passes only (diffuse, specular, light groups...), not data "
               "passes. The Denoise tab decides whether they are denoised too.");
    for (int i = 0; i < 4; ++i)
    {
        static const char* names[] = {"aov1", "aov2", "aov3", "aov4"};
        Input_ChannelSet_knob(f, &m_aovLayer[i], 0, names[i], i == 0 ? "AOVs" : "");
        if (i == 1 || i == 3)
            ClearFlags(f, Knob::STARTLINE);
    }
    String_knob(f, &m_aovPattern, "aov_pattern", "more AOVs");
    Tooltip(f, "More AOV layers by name, separated by spaces. Wildcards work: RGBA_* takes every Arnold "
               "light group, diffuse_* every diffuse pass.");

    Divider(f, "Output");
    Enumeration_knob(f, &m_view, kViewNames, "view", "view");
    Tooltip(f, "Result: the cleaned image.\nFireflies overlay: detections in magenta over the darkened image.\n"
               "Removed: what all the stages took out.\nFirefly matte: the detections.\n"
               "Deflicker weight: how much each pixel was averaged over time (white = fully).");
    Bool_knob(f, &m_outputMatte, "output_matte", "output matte");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Write the firefly matte into the channel on the right.");
    Channel_knob(f, &m_matteChannel, 1, "matte_channel", "");
    ClearFlags(f, Knob::STARTLINE);
    Input_Channel_knob(f, &m_maskChannel, 1, 3, "mask_channel", "mask");
    SetFlags(f, Knob::STARTLINE);
    Tooltip(f, "Channel of the mask input that limits the whole effect.");
    Bool_knob(f, &m_invertMask, "invert_mask", "invert");
    ClearFlags(f, Knob::STARTLINE);
    Float_knob(f, &m_mix, "mix", "mix");
    SetRange(f, 0.0, 1.0);
    SetFlags(f, Knob::STARTLINE);

    Divider(f, "OIDN");
    Enumeration_knob(f, &m_device, kDeviceNames, "device", "device");
    Tooltip(f, "Device used by OIDN. CUDA falls back to CPU when unavailable.");
    Int_knob(f, &m_maxMem, "max_memory", "memory limit (MB)");
    Tooltip(f, "Cap the memory used by OIDN. 0 = no limit.");

    Divider(f, "");
    Text_knob(f, "<b>CleanPass</b> " CP_VERSION " - Gabin Vidal\n"
                 "Uses Intel Open Image Denoise (Apache 2.0).");
}

void CleanPassIop::updateKnobStates()
{
    auto en = [this](const char* n, bool on) {
        if (Knob* k = knob(n))
            k->enable(on);
    };
    for (const char* n : {"reference", "threshold", "isolation", "softness", "noise_adaptive", "min_level",
                          "max_size", "halo", "fix_nan", "temporal", "method", "conserve_energy"})
        en(n, m_ffOn);
    en("motion_radius", m_ffOn && m_temporal);
    en("energy_radius", m_ffOn && m_conserve);
    en("energy_amount", m_ffOn && m_conserve);
    for (const char* n : {"df_frames", "df_amount", "df_tolerance", "df_search", "df_patch"})
        en(n, m_dfOn);
    for (const char* n : {"df_guides", "df_vectors", "df_depth", "df_variance"})
        en(n, m_dfOn || (m_dnOn && m_stabilize));
    for (const char* n : {"denoise_amount", "stabilize"})
        en(n, m_dnOn);
    en("stabilize_tolerance", m_dnOn && m_stabilize);
    en("denoise_aovs", m_dnOn && m_propagate);
    en("aovs_match_beauty", m_dnOn && m_propagate && m_denoiseAovs);
    en("aov_pattern", m_propagate);
    en("quality", m_dnOn || (m_ffOn && m_reference == 0));
    en("prefilter_aux", m_dnOn || (m_ffOn && m_reference == 0));
    for (const char* n : {"aov1", "aov2", "aov3", "aov4"})
        en(n, m_propagate);
    en("matte_channel", m_outputMatte);
    en("albedo_layer", !connected(1));
    en("normal_layer", !connected(2));
    en("mask_channel", connected(3));
    en("invert_mask", connected(3));
}

// Fills the layer knobs from the usual names (all = also the ones already set).
void CleanPassIop::autoLayers(bool all)
{
    Op* op = node_input(0, Op::OUTPUT_OP);
    Iop* src = op ? dynamic_cast<Iop*>(op) : nullptr;
    if (!src)
        return;
    src->validate(false);
    const ChannelSet chans = src->info().channels();
    const char* const knobs[ePassCount] = {"albedo_layer", "normal_layer", "motion_layer", "depth_layer",
                                           "variance_layer"};
    const ChannelSet* values[ePassCount] = {&m_albedoLayer, &m_normalLayer, &m_motionLayer, &m_depthLayer,
                                            &m_varianceLayer};
    for (int p = 0; p < ePassCount; ++p)
    {
        const std::string layer = findPassLayer(chans, p);
        if (!layer.empty() && (all || values[p]->empty()))
            if (Knob* k = knob(knobs[p]))
                k->set_text(layer.c_str());
    }
}

int CleanPassIop::knob_changed(Knob* k)
{
    if (k->is("detect_layers"))
    {
        autoLayers(true);
        return 1;
    }
    if (k->is("inputChange") || k == &Knob::showPanel)
        autoLayers(false);
    if (k == &Knob::showPanel || k->is("inputChange") || k->is("firefly_enable") || k->is("deflicker_enable") ||
        k->is("denoise_enable") || k->is("conserve_energy") || k->is("propagate_aovs") || k->is("temporal") ||
        k->is("output_matte") || k->is("reference") || k->is("stabilize") || k->is("denoise_aovs"))
    {
        updateKnobStates();
        return 1;
    }
    return PlanarIop::knob_changed(k);
}

// ===========================================================================
// validate / requests
// ===========================================================================

std::shared_ptr<const CleanPassIop::Config> CleanPassIop::config() const
{
    cleanpass::LockGuard g(m_cfgLock);
    return m_cfg;
}

void CleanPassIop::_validate(bool for_real)
{
    copy_info(); // from input 0 at the current frame

    // the other inputs and frames are read in renderStripe: validate them now
    for (int n = 1; n < 4; ++n)
        if (connected(n))
            if (Iop* i = in(n))
                i->validate(for_real);
    for (int o = 1; o < std::min(actualCopies(), 1 + 2 * temporalRadius()); ++o)
        if (Iop* t = in(0, o))
            t->validate(for_real);

    m_size = std::max(1, std::min(3, m_size));
    m_grow = std::max(0, std::min(3, m_grow));
    m_motionRadius = std::max(0, std::min(16, m_motionRadius));
    m_maxMem = std::max(0, m_maxMem);
    m_dfSearch = std::max(1, std::min(8, m_dfSearch));
    m_dfPatch = std::max(1, std::min(4, m_dfPatch));

    auto cfg = std::make_shared<Config>();
    Iop* src = in(0);
    const ChannelSet srcChans = src ? src->info().channels() : ChannelSet();

    cfg->albedoInput = connected(1);
    cfg->normalInput = connected(2);
    cfg->maskInput = connected(3) && m_maskChannel != Chan_Black;
    cfg->beauty = rgbOf(m_beautyLayer);
    // an empty layer knob means: the layer with the usual name, if src has one
    auto present = [&](const ChannelSet& cs, int pass) {
        ChannelSet r;
        foreach (z, cs)
            if (srcChans.contains(z))
                r += z;
        if (cs.empty())
        {
            const std::string layer = findPassLayer(srcChans, pass);
            if (!layer.empty())
                foreach (z, srcChans)
                    if (getLayerName(z) == layer)
                        r += z;
        }
        return rgbOf(r);
    };
    cfg->albedo = cfg->albedoInput ? rgbOf(Mask_RGB) : present(m_albedoLayer, eAlbedo);
    cfg->normal = cfg->normalInput ? rgbOf(Mask_RGB) : present(m_normalLayer, eNormal);
    cfg->motion = present(m_motionLayer, eMotion);
    cfg->depth = present(m_depthLayer, eDepth);
    cfg->variance = present(m_varianceLayer, eVariance);

    // AOV layers: one entry per layer, never the beauty, the guides or the data passes
    if (m_propagate && (m_ffOn || m_dfOn || m_dnOn))
    {
        ChannelSet excluded = cfg->beauty.set;
        if (!cfg->albedoInput)
            excluded += cfg->albedo.set;
        if (!cfg->normalInput)
            excluded += cfg->normal.set;
        excluded += cfg->motion.set;
        excluded += cfg->depth.set;
        excluded += cfg->variance.set;
        auto addLayer = [&](const std::string& layer) {
            for (const Rgb& r : cfg->aovs)
                if (r.name == layer)
                    return;
            ChannelSet layerSet;
            foreach (w, srcChans)
                if (getLayerName(w) == layer && !excluded.contains(w) && colourIndex(w) <= 2)
                    layerSet += w;
            Rgb r = rgbOf(layerSet);
            if (r.valid())
            {
                r.name = layer;
                cfg->aovs.push_back(r);
            }
        };
        for (const ChannelSet& cs : m_aovLayer)
            foreach (z, cs)
                if (!excluded.contains(z) && srcChans.contains(z) && colourIndex(z) <= 2)
                    addLayer(getLayerName(z));
        // names / wildcards
        if (m_aovPattern && *m_aovPattern)
        {
            std::vector<std::string> pats;
            std::string cur;
            for (const char* c = m_aovPattern;; ++c)
            {
                if (!*c || std::isspace((unsigned char)*c) || *c == ',')
                {
                    if (!cur.empty())
                        pats.push_back(cur);
                    cur.clear();
                    if (!*c)
                        break;
                }
                else
                    cur += *c;
            }
            std::vector<std::string> layers;
            foreach (w, srcChans)
            {
                const std::string l = getLayerName(w);
                if (std::find(layers.begin(), layers.end(), l) == layers.end())
                    layers.push_back(l);
            }
            for (const std::string& l : layers)
                for (const std::string& pat : pats)
                    if (globMatch(pat.c_str(), l.c_str()))
                    {
                        addLayer(l);
                        break;
                    }
        }
    }

    // stage parameters
    cleanpass::PipelineParams& pp = cfg->pipe;
    pp.firefly = m_ffOn;
    pp.ff.reference = (cleanpass::ReferenceMode)std::max(0, std::min(1, m_reference));
    pp.ff.threshold = m_threshold;
    pp.ff.softness = m_softness;
    pp.ff.isolation = m_isolation;
    pp.ff.adaptivity = m_adaptivity;
    pp.ff.minLevel = m_minLevel;
    pp.ff.size = m_size;
    pp.ff.grow = m_grow;
    pp.ff.fixNonFinite = m_fixNonFinite;
    pp.ff.method = (cleanpass::Method)std::max(0, std::min(2, m_method));
    pp.ff.conserveEnergy = m_conserve;
    pp.ff.energyRadius = m_energyRadius;
    pp.ff.energyAmount = m_energyAmount;
    pp.ff.temporal = m_temporal;
    pp.ff.motionRadius = m_motionRadius;
    pp.deflicker = m_dfOn;
    pp.df.frames = m_dfFrames >= 1 ? 2 : 1;
    pp.df.search = m_dfSearch;
    pp.df.patch = m_dfPatch;
    pp.df.tolerance = std::max(0.1f, m_dfTolerance);
    pp.df.amount = std::max(0.f, std::min(1.f, m_dfAmount));
    pp.df.useGuides = m_dfGuides;
    pp.df.useVectors = m_dfVectors;
    pp.df.useDepth = m_dfDepth;
    pp.df.useVariance = m_dfVariance;
    pp.denoise = m_dnOn;
    pp.denoiseAmount = std::max(0.f, std::min(1.f, m_dnAmount));
    pp.stabilize = m_stabilize;
    pp.stabilizeTolerance = std::max(0.001f, m_stabTol);
    pp.denoiseAovs = m_denoiseAovs;
    pp.aovsMatchBeauty = m_aovsMatch;

    cfg->oidn.device = m_device;
    cfg->oidn.quality = m_quality;
    cfg->oidn.prefilterAux = m_prefilterAux;
    cfg->oidn.maxMemoryMB = m_maxMem;
    cfg->maskChannel = m_maskChannel;
    cfg->invertMask = m_invertMask;
    cfg->view = m_view;
    cfg->mix = std::max(0.f, std::min(1.f, m_mix));
    cfg->outputMatte = m_outputMatte && m_matteChannel != Chan_Black;
    cfg->matteChannel = m_matteChannel;
    cfg->copies = std::min(actualCopies(), 1 + 2 * temporalRadius());

    if (cfg->outputMatte)
        info_.turn_on(cfg->matteChannel);

    cfg->box = info();
    cfg->box.intersect(info().format());

    if (!cfg->active() || (cfg->mix <= 0.f && !cfg->outputMatte && cfg->view == eResult))
        set_out_channels(Mask_None);
    else
        set_out_channels(Mask_All);

    cleanpass::LockGuard g(m_cfgLock);
    m_cfg = cfg;
}

void CleanPassIop::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& req) const
{
    Iop* src = in(0);
    if (!src)
        return;
    auto clip = [](const Box& b, Iop* iop) {
        Box r = b;
        r.intersect(bboxOf(iop));
        return r;
    };
    req.request(src, clip(box, src), channels, count);

    std::shared_ptr<const Config> cfg = config();
    if (!cfg || !cfg->active() || cfg->box.w() <= 0 || cfg->box.h() <= 0)
        return;
    ChannelSet need = cfg->beauty.set;
    if (!cfg->albedoInput)
        need += cfg->albedo.set;
    if (!cfg->normalInput)
        need += cfg->normal.set;
    for (const Rgb& r : cfg->aovs)
        need += r.set;
    need += cfg->motion.set;
    need += cfg->depth.set;
    need += cfg->variance.set;
    req.request(src, clip(cfg->box, src), need, count);
    for (int o = 1; o < cfg->copies; ++o)
        if (Iop* t = in(0, o))
            req.request(t, clip(cfg->box, t), need, count);

    if (cfg->albedoInput)
        if (Iop* a = in(1))
            req.request(a, clip(cfg->box, a), Mask_RGB, count);
    if (cfg->normalInput)
        if (Iop* n = in(2))
            req.request(n, clip(cfg->box, n), Mask_RGB, count);
    if (cfg->maskInput)
        if (Iop* m = in(3))
            req.request(m, clip(cfg->box, m), ChannelSet(cfg->maskChannel), count);
}

// ===========================================================================
// analysis (cached per frame / settings)
// ===========================================================================

Hash CleanPassIop::analysisKey(const Config& cfg) const
{
    // Everything the analysis depends on - but not mix / view / matte output,
    // so that those can be changed without recomputing anything.
    Hash h;
    h.append(outputContext().frame()); // some ops keep one hash on every frame
    for (int n = 0; n < 4; ++n)
    {
        const bool c = (n == 0) || (n == 1 && cfg.albedoInput) || (n == 2 && cfg.normalInput) ||
                       (n == 3 && cfg.maskInput);
        h.append(c);
        if (c)
            if (Iop* i = in(n))
                h.append(i->hash());
    }
    for (int o = 1; o < cfg.copies; ++o)
        if (Iop* t = in(0, o))
            h.append(t->hash());
    h.append(settingsKey(cfg));
    return h;
}

Hash CleanPassIop::settingsKey(const Config& cfg) const
{
    Hash h;
    const cleanpass::PipelineParams& pp = cfg.pipe;
    const cleanpass::Params& p = pp.ff;
    h.append(pp.firefly);
    h.append((int)p.reference);
    h.append(p.threshold); h.append(p.softness); h.append(p.isolation); h.append(p.adaptivity);
    h.append(p.minLevel); h.append(p.size); h.append(p.grow); h.append(p.fixNonFinite);
    h.append((int)p.method); h.append(p.conserveEnergy); h.append(p.energyRadius); h.append(p.energyAmount);
    h.append(p.temporal); h.append(p.motionRadius);
    h.append(pp.deflicker);
    h.append(pp.df.frames); h.append(pp.df.search); h.append(pp.df.patch); h.append(pp.df.tolerance);
    h.append(pp.df.amount); h.append(pp.df.useGuides);
    h.append(pp.denoise); h.append(pp.denoiseAmount);
    h.append(pp.df.useVectors); h.append(pp.df.useDepth); h.append(pp.df.useVariance);
    h.append(pp.stabilize); h.append(pp.stabilizeTolerance); h.append(pp.denoiseAovs); h.append(pp.aovsMatchBeauty);
    h.append(cfg.oidn.device); h.append(cfg.oidn.quality); h.append(cfg.oidn.prefilterAux);
    h.append(cfg.oidn.maxMemoryMB);
    h.append((int)cfg.maskChannel); h.append(cfg.invertMask);
    for (const Rgb* r : {&cfg.beauty, &cfg.albedo, &cfg.normal, &cfg.motion, &cfg.depth, &cfg.variance})
        for (int c = 0; c < 3; ++c)
            h.append((int)r->ch[c]);
    for (const Rgb& r : cfg.aovs)
        for (int c = 0; c < 3; ++c)
            h.append((int)r.ch[c]);
    h.append(cfg.box.x()); h.append(cfg.box.y()); h.append(cfg.box.r()); h.append(cfg.box.t());
    return h;
}

std::shared_ptr<CleanPassIop::Result> CleanPassIop::analysis(const Config& cfg)
{
    const Hash key = analysisKey(cfg);
    cleanpass::LockGuard g(m_lock); // one analysis at a time per node; other planes wait for it
    if (m_cache && m_cacheKey == key)
        return m_cache;
    std::shared_ptr<Result> a;
    try
    {
        a = computeAnalysis(cfg);
    }
    catch (const std::exception& e)
    {
        error("CleanPass: %s", e.what());
        a.reset();
    }
    catch (...)
    {
        error("CleanPass: unexpected error");
        a.reset();
    }
    if (a && a->r.valid)
    {
        m_cache = a;
        m_cacheKey = key;
    }
    return a;
}

std::shared_ptr<const cleanpass::PipelineResult> CleanPassIop::denoisedGet(const Hash& k)
{
    cleanpass::LockGuard g(m_dnLock);
    for (const Denoised& d : m_dnCache)
        if (d.key == k)
            return d.res;
    return nullptr;
}

namespace
{
bool isFull(const cleanpass::PipelineResult& r) { return !r.beautyStage2.empty(); }
// only the denoised image: what a neighbour's stabilisation needs
std::shared_ptr<const cleanpass::PipelineResult> slim(const cleanpass::PipelineResult& r, const std::vector<float>& img)
{
    auto s = std::make_shared<cleanpass::PipelineResult>();
    s->width = r.width;
    s->height = r.height;
    s->beauty = img;
    s->valid = true;
    return s;
}
} // namespace

void CleanPassIop::denoisedPut(const Hash& k, std::shared_ptr<const cleanpass::PipelineResult> r, bool replace)
{
    if (!r || !r->valid || r->beauty.empty())
        return;
    cleanpass::LockGuard g(m_dnLock);
    for (Denoised& d : m_dnCache)
        if (d.key == k)
        {
            if (replace)
                d.res = r;
            return;
        }
    m_dnCache.push_back(Denoised{k, r});
    while (m_dnCache.size() > 4) // t-1, t, t+1 and one spare
        m_dnCache.erase(m_dnCache.begin());
    // full entries are heavy (analysis of the whole frame): keep two at most
    int full = 0;
    for (auto it = m_dnCache.rbegin(); it != m_dnCache.rend(); ++it)
        if (isFull(*it->res) && ++full > 2)
            it->res = slim(*it->res, it->res->beauty);
}

std::shared_ptr<CleanPassIop::Result> CleanPassIop::computeAnalysis(const Config& cfg)
{
    auto res = std::make_shared<Result>();
    Iop* src = in(0);
    const Box box = cfg.box;
    if (!src || box.w() <= 0 || box.h() <= 0 || !cfg.beauty.valid())
        return res;
    auto stop = [this]() { return aborted() || cancelled(); };
    const cleanpass::PipelineParams& pp = cfg.pipe;
    const bool stab = pp.denoise && pp.stabilize && pp.denoiseAmount > 0.f;
    const bool temporalMotion = pp.deflicker || stab;

    cleanpass::PipelineInput inp;
    inp.width = box.w();
    inp.height = box.h();

    // current frame
    std::vector<float> beauty, albedo, normal, mask, motion, depth, variance;
    fetchDense(src, box, cfg.beauty.ch, beauty);
    inp.cur.beauty = viewOf(beauty, box);
    if (cfg.albedo.valid())
    {
        fetchDense(cfg.albedoInput ? in(1) : src, box, cfg.albedo.ch, albedo);
        inp.cur.albedo = viewOf(albedo, box);
    }
    if (cfg.normal.valid())
    {
        fetchDense(cfg.normalInput ? in(2) : src, box, cfg.normal.ch, normal);
        inp.cur.normal = viewOf(normal, box);
    }
    const bool useMotion = temporalMotion && pp.df.useVectors && cfg.motion.valid();
    const bool useDepth = temporalMotion && pp.df.useDepth && cfg.depth.valid();
    const bool useVariance = temporalMotion && pp.df.useVariance && cfg.variance.valid();
    if (useMotion)
    {
        fetchDense(src, box, cfg.motion.ch, motion);
        inp.cur.motion = viewOf(motion, box);
    }
    if (useDepth)
    {
        fetchDense(src, box, cfg.depth.ch, depth);
        inp.cur.depth = viewOf(depth, box);
    }
    if (useVariance)
    {
        fetchDense(src, box, cfg.variance.ch, variance);
        inp.cur.variance = viewOf(variance, box);
    }
    if (stop())
        return res;

    // neighbour frames
    const int need = std::max({pp.firefly && pp.ff.temporal ? 1 : 0, pp.deflicker ? pp.df.frames : 0, stab ? 2 : 0});
    const bool guidesFromSrc = ((temporalMotion && pp.df.useGuides) || stab) && !cfg.albedoInput &&
                               !cfg.normalInput && cfg.albedo.valid() && cfg.normal.valid();
    // The stabilisation denoises t-1 / t+1 too: it needs their guides, which a
    // separate albedo / normal input only gives at frame t.
    cleanpass::PipelineParams runP = pp;
    if (stab && (cfg.albedoInput || cfg.normalInput))
        runP.stabilize = false;
    std::vector<std::vector<float>> nbBeauty(5), nbAlb(5), nbNrm(5), nbMot(5), nbDep(5), nbVar(5);
    Iop* t0 = in(0, 0);
    for (int o = 1; o < cfg.copies; ++o)
    {
        const int off = offsetOfCopy(o);
        if (std::abs(off) > need)
            continue;
        Iop* t = in(0, o);
        if (!t || !t0)
            continue;
        fetchDense(t, box, cfg.beauty.ch, nbBeauty[o]);
        // A neighbour identical to the current frame (still image, held
        // frame) brings nothing and would make every firefly look "real".
        // Compared on the pixels: some ops (an Expression using `frame`) keep
        // the same hash on every frame.
        if (nbBeauty[o].size() == beauty.size() &&
            std::memcmp(nbBeauty[o].data(), beauty.data(), beauty.size() * sizeof(float)) == 0)
        {
            std::vector<float>().swap(nbBeauty[o]);
            continue;
        }
        cleanpass::PipelineFrame fr;
        fr.offset = off;
        fr.beauty = viewOf(nbBeauty[o], box);
        if (guidesFromSrc)
        {
            fetchDense(t, box, cfg.albedo.ch, nbAlb[o]);
            fetchDense(t, box, cfg.normal.ch, nbNrm[o]);
            fr.albedo = viewOf(nbAlb[o], box);
            fr.normal = viewOf(nbNrm[o], box);
        }
        if (useDepth)
        {
            fetchDense(t, box, cfg.depth.ch, nbDep[o]);
            fr.depth = viewOf(nbDep[o], box);
        }
        // t-1 / t+1 are processed on their own by the stabilisation: their own
        // vectors and variance are needed then
        if (stab && std::abs(off) == 1)
        {
            if (useMotion)
            {
                fetchDense(t, box, cfg.motion.ch, nbMot[o]);
                fr.motion = viewOf(nbMot[o], box);
            }
            if (useVariance)
            {
                fetchDense(t, box, cfg.variance.ch, nbVar[o]);
                fr.variance = viewOf(nbVar[o], box);
            }
        }
        inp.neighbours.push_back(fr);
        res->nbCopy.push_back(o);
        if (stop())
            return res;
    }

    // global mask (applied at output, to every stage)
    if (cfg.maskInput)
        if (Iop* mi = in(3))
        {
            const Channel mc[3] = {cfg.maskChannel, Chan_Black, Chan_Black};
            fetchDense(mi, box, mc, mask);
            mask.resize((size_t)box.w() * box.h());
            for (float& v : mask)
            {
                v = std::isfinite(v) ? std::max(0.f, std::min(1.f, v)) : 0.f;
                if (cfg.invertMask)
                    v = 1.f - v;
            }
            res->mask = mask;
        }
    if (stop())
        return res;

    // stabilisation: denoised results of t-1 / t+1 already computed (when
    // playing or rendering in order, frame t+1's own result is one of them)
    auto copyHash = [&](int o) {
        Iop* t = in(0, o);
        return t ? t->hash() : Hash();
    };
    auto denoisedKey = [&](int center, int minus, int plus) {
        Hash h;
        h.append(settingsKey(cfg));
        h.append(outputContext().frame() + offsetOfCopy(center));
        h.append(copyHash(center));
        h.append(copyHash(minus));
        h.append(copyHash(plus));
        return h;
    };
    const bool cacheable = runP.stabilize && stab && !cfg.maskInput && cfg.copies >= 5;
    // the result made for this frame by a neighbour's stabilisation is this
    // frame's own (stages 1 - 3), when the chain is the same: deflicker on 1 frame
    const bool sameChain = !pp.deflicker || pp.df.frames == 1;
    std::shared_ptr<const cleanpass::PipelineResult> dPrev, dNext, dCur;
    Hash kPrev, kNext, kCur;
    if (cacheable)
    {
        kPrev = denoisedKey(1, 3, 0);
        kNext = denoisedKey(2, 0, 4);
        kCur = denoisedKey(0, 1, 2);
        dPrev = denoisedGet(kPrev);
        dNext = denoisedGet(kNext);
        inp.denoisedPrev = dPrev ? &dPrev->beauty : nullptr;
        inp.denoisedNext = dNext ? &dNext->beauty : nullptr;
        if (sameChain)
        {
            dCur = denoisedGet(kCur);
            if (dCur && isFull(*dCur))
                inp.precomputed = dCur.get();
        }
    }

    const cleanpass::OidnSettings os = cfg.oidn;
    const std::function<bool()> isAborted = stop;
    cleanpass::ReferenceFn oidn = [&](int w, int h, const float* color, const cleanpass::View3& alb, const cleanpass::View3& nrm,
                                float* out, std::string& err) {
        return cleanpass::oidnPredict(os, w, h, color, alb, nrm, out, err, isAborted);
    };

    std::string err;
    if (!cleanpass::runPipeline(inp, runP, oidn, oidn, res->r, err, isAborted))
    {
        if (!stop())
            error("CleanPass: %s", err.c_str());
        res->r.valid = false;
        return res;
    }
    if (!res->r.warning.empty())
        warning("CleanPass: %s", res->r.warning.c_str());

    // AOVs: the firefly stage needs every selected AOV at frame t to share the
    // removed energy between them
    if (pp.firefly && !cfg.aovs.empty())
    {
        cleanpass::beginAovs(res->r.firefly, (int)cfg.aovs.size());
        std::vector<float> layer;
        const size_t n = (size_t)box.w() * box.h();
        for (size_t j = 0; j < cfg.aovs.size(); ++j)
        {
            if (stop())
            {
                res->r.valid = false;
                return res;
            }
            fetchDense(src, box, cfg.aovs[j].ch, layer);
            for (int c = 0; c < 3; ++c)
                cleanpass::addAovChannel(res->r.firefly, pp.ff, (int)j, c, layer.data() + c * n, box.w());
        }
        cleanpass::finalizeAovs(res->r.firefly);
    }
    if (stop())
    {
        res->r.valid = false; // some input planes may be missing: never cache this
        return res;
    }

    // keep what the next frames can reuse
    if (cacheable)
    {
        // the frames the stabilisation processed: complete when they used this
        // frame's chain, so that they skip their own stages 1 - 3 later
        if (res->r.subPrev)
            denoisedPut(kPrev, sameChain ? std::shared_ptr<const cleanpass::PipelineResult>(res->r.subPrev)
                                         : slim(*res->r.subPrev, res->r.subPrev->beauty));
        if (res->r.subNext)
            denoisedPut(kNext, sameChain ? std::shared_ptr<const cleanpass::PipelineResult>(res->r.subNext)
                                         : slim(*res->r.subNext, res->r.subNext->beauty));
        // this frame's own denoised result is what t-1 / t+1 need, if it was
        // made with the same chain (deflicker on +-1 frame only); it replaces
        // the complete entry this frame may have used (no longer needed)
        if (res->r.stabilized && sameChain)
            denoisedPut(kCur, slim(res->r, res->r.beautyDenoised), true);
    }
    // the cache owns them now (or they are not needed): do not keep them with this frame
    res->r.subPrev.reset();
    res->r.subNext.reset();
    res->albedo.swap(albedo);
    res->normal.swap(normal);

    const cleanpass::DeflickerResult& D = res->r.deflicker;
    std::cerr << "[CleanPass] frame " << outputContext().frame() << ":";
    if (res->r.reused)
        std::cerr << " [firefly / deflicker / denoise reused from the previous frame]";
    if (pp.firefly)
        std::cerr << " firefly " << res->r.firefly.numCore << " (" << res->r.seconds[0] << " s)";
    if (pp.deflicker)
        std::cerr << " | deflicker " << D.count << " frames (" << res->r.seconds[1] << " s)";
    if (temporalMotion && D.valid)
    {
        if (D.vectorsUsed)
            std::cerr << " [vectors x" << D.vecX << " y" << D.vecY << ", " << (int)(100.f * D.vectorShare + 0.5f)
                      << "% of matches]";
        else if (useMotion)
            std::cerr << " [vectors ignored: they do not match the image]";
        if (D.fixedSeed)
            std::cerr << " [fixed render seed detected]";
        if (D.varianceUsed)
            std::cerr << " [variance]";
        if (D.depthUsed)
            std::cerr << " [depth]";
    }
    if (pp.denoise)
        std::cerr << " | denoise (" << res->r.seconds[2] << " s" << (res->r.stabilized ? ", stabilised" : "")
                  << ((dPrev || dNext) ? ", neighbours reused" : "") << ")";
    std::cerr << std::endl;
    return res;
}

void CleanPassIop::computeAovs(const Config& cfg, Result& res)
{
    cleanpass::LockGuard g(res.aovLock);
    if (res.aovDone)
        return;
    Iop* src = in(0);
    const Box box = cfg.box;
    const size_t N = (size_t)box.w() * box.h();
    const size_t na = cfg.aovs.size();
    if (!src || na == 0)
        return;
    auto stop = [this]() { return aborted() || cancelled(); };
    const cleanpass::PipelineResult& R = res.r;

    std::vector<std::vector<float>> cur(na), nbData(na * res.nbCopy.size());
    std::vector<int> layers(na);
    std::vector<std::array<const float*, 3>> curP(na);
    std::vector<std::vector<std::array<const float*, 3>>> nbP(na);
    std::vector<std::vector<float>> out(na, std::vector<float>(N * 3, 0.f));
    std::vector<std::array<float*, 3>> outP(na);
    for (size_t j = 0; j < na; ++j)
    {
        if (stop())
            return;
        fetchDense(src, box, cfg.aovs[j].ch, cur[j]);
        curP[j] = {cur[j].data(), cur[j].data() + N, cur[j].data() + 2 * N};
        outP[j] = {out[j].data(), out[j].data() + N, out[j].data() + 2 * N};
        layers[j] = cfg.pipe.firefly ? (int)j : -2;
        nbP[j].resize(res.nbCopy.size());
        for (size_t q = 0; q < res.nbCopy.size(); ++q)
        {
            nbP[j][q] = {nullptr, nullptr, nullptr};
            if (!(cfg.pipe.deflicker && R.deflicker.valid))
                continue;
            std::vector<float>& d = nbData[j * res.nbCopy.size() + q];
            if (Iop* t = in(0, res.nbCopy[q]))
                fetchDense(t, box, cfg.aovs[j].ch, d);
            if (!d.empty())
                nbP[j][q] = {d.data(), d.data() + N, d.data() + 2 * N};
        }
    }
    if (stop())
        return;
    cleanpass::PipelineInput gin;
    gin.width = box.w();
    gin.height = box.h();
    gin.cur.albedo = viewOf(res.albedo, box);
    gin.cur.normal = viewOf(res.normal, box);
    const cleanpass::OidnSettings os = cfg.oidn;
    const std::function<bool()> isAborted = stop;
    cleanpass::ReferenceFn oidn = [&](int w, int h, const float* color, const cleanpass::View3& alb, const cleanpass::View3& nrm,
                                float* o, std::string& err) {
        return cleanpass::oidnPredict(os, w, h, color, alb, nrm, o, err, isAborted);
    };
    const std::string w = cleanpass::pipelineAovs(R, cfg.pipe, gin, oidn, layers, curP, nbP, outP);
    if (stop())
        return;
    if (!w.empty())
        std::cerr << "[CleanPass] " << w << std::endl;
    res.aovOut.swap(out);
    res.aovDone = true;
}

// ===========================================================================
// render
// ===========================================================================

void CleanPassIop::renderStripe(ImagePlane& plane)
{
    plane.makeWritable();
    const Box pb = plane.bounds();
    const ChannelSet pc = plane.channels();
    const int nc = plane.nComps();
    float* const base = plane.writable();
    const int64_t cs = plane.chanStride(), rs = plane.rowStride(), ps = plane.colStride();
    auto px = [&](int ci, int x, int y) -> float& {
        return base[ci * cs + (int64_t)(y - pb.y()) * rs + (int64_t)(x - pb.x()) * ps];
    };

    // 1. pass-through of the input (zero where the input has nothing)
    for (int i = 0; i < nc; ++i)
        for (int y = pb.y(); y < pb.t(); ++y)
            for (int x = pb.x(); x < pb.r(); ++x)
                px(i, x, y) = 0.f;
    Iop* src = in(0);
    if (src)
    {
        const ChannelSet have = pc.intersection(src->info().channels());
        Box inter = pb;
        inter.intersect(bboxOf(src));
        if (!have.empty() && inter.w() > 0 && inter.h() > 0)
        {
            Row row(inter.x(), inter.r());
            for (int y = inter.y(); y < inter.t(); ++y)
            {
                if (aborted() || cancelled())
                    return;
                src->get(y, inter.x(), inter.r(), have, row);
                foreach (z, have)
                {
                    const int di = plane.chanNo(z);
                    const float* s = row[z];
                    if (di < 0 || di >= nc || !s)
                        continue;
                    for (int x = inter.x(); x < inter.r(); ++x)
                        px(di, x, y) = s[x];
                }
            }
        }
    }
    if (aborted() || cancelled())
        return;

    std::shared_ptr<const Config> cfgPtr = config();
    if (!cfgPtr || !cfgPtr->active())
        return;
    const Config& cfg = *cfgPtr;

    // 2. does this plane hold anything we change?
    auto indicesOf = [&](const Rgb& r, int idx[3]) {
        bool any = false;
        for (int c = 0; c < 3; ++c)
        {
            idx[c] = -1;
            bool dup = false; // single-channel layers: write the channel once
            for (int d = 0; d < c; ++d)
                dup = dup || r.ch[d] == r.ch[c];
            if (!dup && r.ch[c] != Chan_Black && pc.contains(r.ch[c]))
            {
                const int i = plane.chanNo(r.ch[c]);
                if (i >= 0 && i < nc)
                {
                    idx[c] = i;
                    any = true;
                }
            }
        }
        return any;
    };
    int beautyIdx[3];
    const bool hasBeauty = indicesOf(cfg.beauty, beautyIdx);
    int aovLayer = -1, aovIdx[3] = {-1, -1, -1};
    for (size_t j = 0; j < cfg.aovs.size() && aovLayer < 0; ++j)
        if (indicesOf(cfg.aovs[j], aovIdx))
            aovLayer = (int)j;
    int matteIdx = -1;
    if (cfg.outputMatte && pc.contains(cfg.matteChannel))
    {
        matteIdx = plane.chanNo(cfg.matteChannel);
        if (matteIdx >= nc)
            matteIdx = -1;
    }
    if (!hasBeauty && aovLayer < 0 && matteIdx < 0)
        return;

    std::shared_ptr<Result> res = analysis(cfg);
    if (!res || !res->r.valid || aborted() || cancelled())
        return;
    const cleanpass::PipelineResult& R = res->r;
    const int W = cfg.box.w(), H = cfg.box.h();
    if (R.width != W || R.height != H)
        return; // stale, should not happen
    const size_t N = (size_t)W * H;

    // area of the plane covered by the processed box
    Box area = pb;
    area.intersect(cfg.box);
    if (area.w() <= 0 || area.h() <= 0)
        return;
    auto amountAt = [&](size_t i) {
        float k = cfg.mix;
        if (!res->mask.empty())
            k *= res->mask[i];
        return k;
    };
    auto blend = [](float in, float out, float k) {
        if (k <= 0.f)
            return in;
        if (!std::isfinite(in))
            return out;
        return in + k * (out - in);
    };

    // 3. beauty
    if (hasBeauty)
    {
        const float overlay[3] = {1.f, 0.f, 1.f};
        for (int y = area.y(); y < area.t(); ++y)
            for (int x = area.x(); x < area.r(); ++x)
            {
                const size_t i = (size_t)(y - cfg.box.y()) * W + (x - cfg.box.x());
                const float k = amountAt(i);
                const float m = R.firefly.matte.empty() ? 0.f : R.firefly.matte[i];
                float dw = 0.f;
                if (cfg.view == eDfWeight && R.deflicker.valid && R.deflicker.count > 0)
                {
                    for (int q = 0; q < R.deflicker.count; ++q)
                        dw += R.deflicker.weight[N * q + i];
                    dw /= (float)R.deflicker.count;
                }
                for (int c = 0; c < 3; ++c)
                {
                    if (beautyIdx[c] < 0)
                        continue;
                    float& o = px(beautyIdx[c], x, y);
                    const float in = o;
                    const float result = blend(in, R.beauty[c * N + i], k);
                    switch (cfg.view)
                    {
                    case eResult: o = result; break;
                    case eRemoved: o = (std::isfinite(in) ? in : 0.f) - result; break;
                    case eOverlay: o = (std::isfinite(result) ? 0.35f * result : 0.f) + m * overlay[c] * 4.f; break;
                    case eMatte: o = m; break;
                    case eDfWeight: o = dw; break;
                    default: o = result; break;
                    }
                }
            }
    }

    // 4. AOVs: every stage, all AOVs computed together (they share the corrections)
    if (aovLayer >= 0 && cfg.mix > 0.f)
    {
        computeAovs(cfg, *res);
        if (!res->aovDone || aovLayer >= (int)res->aovOut.size())
            return;
        const std::vector<float>& outBuf = res->aovOut[aovLayer];
        for (int y = area.y(); y < area.t(); ++y)
            for (int x = area.x(); x < area.r(); ++x)
            {
                const size_t i = (size_t)(y - cfg.box.y()) * W + (x - cfg.box.x());
                const float k = amountAt(i);
                for (int c = 0; c < 3; ++c)
                    if (aovIdx[c] >= 0)
                    {
                        float& o = px(aovIdx[c], x, y);
                        o = blend(o, outBuf[c * N + i], k);
                    }
            }
    }

    // 5. firefly matte channel
    if (matteIdx >= 0)
        for (int y = pb.y(); y < pb.t(); ++y)
            for (int x = pb.x(); x < pb.r(); ++x)
            {
                float m = 0.f;
                if (x >= cfg.box.x() && x < cfg.box.r() && y >= cfg.box.y() && y < cfg.box.t() &&
                    !R.firefly.matte.empty())
                {
                    const size_t i = (size_t)(y - cfg.box.y()) * W + (x - cfg.box.x());
                    m = R.firefly.matte[i] * amountAt(i);
                }
                px(matteIdx, x, y) = m;
            }
}

// returns Op*: Nuke 16 wants a plain Op*(*)(Node*) constructor
static Op* build(Node* node) { return new CleanPassIop(node); }
const Iop::Description CleanPassIop::d(kClass, build);

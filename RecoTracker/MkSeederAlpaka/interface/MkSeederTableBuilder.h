#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederTableBuilder_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederTableBuilder_h

// The seeder's static tables from its configuration and the tracker (host code, MkSeederESProducer): the binned
// layers and their axes, the chains of both z sides (crossing envelopes, start pairs with their cot ranges and hole
// positions, the window tables and the tables of the layer combinations), the chains' constants, the shape bands, the
// gap map and the patterns' order. And the per-event host helpers of the module: the hits' bin keys and the gap map's
// ray tables for the event's beam spot.

#include <cstdint>
#include <vector>

#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederESData.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederHostTables.h"

namespace mkseeder {

  // The tracker as the table builder needs it (from mkfit::TrackerInfo, by the ESProducer), by mkFit layer id
  struct MkSeederTrackerLayer {
    bool isPixel = false, isBarrel = false;
    float rIn = 0, rOut = 0, zMin = 0, zMax = 0;
    int nModules = 0;
  };
  // a barrel module for the gap map (mkfit::ModuleInfo and its ModuleShape)
  struct MkSeederGapMapModule {
    float position[3], zDirection[3], xDirection[3];
    float yDirectionZ;            // the z component of the module's y direction, zDirection x xDirection
    float halfWidth, halfLength;  // ModuleShape::dx1, dy
    bool isRectangle;             // ModuleShape::is_rect()
  };
  struct MkSeederTrackerDescription {
    std::vector<MkSeederTrackerLayer> layers;
    std::vector<std::vector<MkSeederGapMapModule>> modules;  // per layer its modules (barrel layers only)
  };

  // Fills the tables' layer geometry (without the modules' positions: firstModule only), chains, chain parameters,
  // shapes, gap map and patterns, and the host tables. qBinScale multiplies the q bin width of the binned layers
  // (2 cm barrel, 1 cm disc).
  void buildMkSeederTables(const MkSeederConfig &config,
                           const MkSeederTrackerDescription &tracker,
                           double qBinScale,
                           MkSeederESDataHost &tables,
                           MkSeederHostTables &hostTables);

  // the event's beam spot (the hits' transverse coordinates are relative to its line at the hit's z)
  struct MkSeederBeamSpot {
    float x = 0, y = 0, z = 0, dxdz = 0, dydz = 0;
  };
  // The hits [0, n) of a binned layer (global x, y, z in): their beam-relative phi, r, 1/r, x, y and their sort key
  // (q fine bin << 24 | phi fine bin); extent (qbar low, qbar high, q low, q high) widened by them
  struct MkSeederHitBinningOutput {
    float *phi, *r, *invR, *xBeam, *yBeam;
    uint32_t *key;
  };
  void computeHitBinning(const MkSeederLayerBinning &layer,
                         const MkSeederBeamSpot &beamSpot,
                         uint32_t n,
                         const float *x,
                         const float *y,
                         const float *z,
                         const MkSeederHitBinningOutput &out,
                         double extent[4]);

  // The gap map's ray table of gap-map layer lid for rays from (bx, by): per phi bin the radii at which the ray
  // crosses the layer's ladder planes (rInner <= rOuter) and a flag where it passes within tolerance of a ladder's
  // long edge or crosses none
  void computeGapRays(const MkSeederGapMapSoA::ConstView &gapMap,
                      int lid,
                      float bx,
                      float by,
                      int nPhi,
                      float tolerance,
                      float *rInner,
                      float *rOuter,
                      unsigned char *nearEdge);

}  // namespace mkseeder

#endif

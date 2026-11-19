#include <stdexcept>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"

namespace mkseeder {

  PSMarkGeometry buildPSMarkGeometry(const mkfit::TrackerInfo &ti, const TrackerTopology &topo) {
    PSMarkGeometry g;
    constexpr int kP[3] = {4, 6, 8};
    for (int i = 0; i < 3; ++i) {
      PSMarkLayer &L = g.layers[i];
      L.pLayer = kP[i], L.sLayer = kP[i] + 1;
      if (L.sLayer >= ti.n_layers())
        continue;
      const auto &lp = ti[L.pLayer];
      const auto &ls = ti[L.sLayer];
      const int n = lp.n_modules();
      L.partner.assign(n, -1), L.frame.assign(n, {}), L.separation.assign(n, 0.f);
      for (int m = 0; m < n; ++m) {
        const auto &mi = lp.module_info(m);
        // the frame: n outwards, u along +phi, v = n x u
        float nx = mi.zdir[0], ny = mi.zdir[1], nz = mi.zdir[2];
        float ux = mi.xdir[0], uy = mi.xdir[1], uz = mi.xdir[2];
        const float px = mi.pos[0], py = mi.pos[1], pz = mi.pos[2];
        if (nx * px + ny * py < 0)
          nx = -nx, ny = -ny, nz = -nz;
        if (-ux * py + uy * px < 0)
          ux = -ux, uy = -uy, uz = -uz;
        const float vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
        L.frame[m] = {ux, uy, uz, vx, vy, vz, nx, ny, nz};
        const DetId partner = topo.partnerDetId(DetId(mi.detid));
        if (partner.rawId() == 0)
          continue;
        int sid = -1;
        try {
          sid = int(ls.short_id(partner.rawId()));
        } catch (const std::out_of_range &) {
          sid = -1;
        }
        if (sid < 0 || sid >= ls.n_modules())
          continue;
        const auto &si = ls.module_info(sid);
        L.partner[m] = sid;
        L.separation[m] = (si.pos[0] - px) * nx + (si.pos[1] - py) * ny + (si.pos[2] - pz) * nz;
        ++g.nPartnered;
      }
    }
    return g;
  }

}  // namespace mkseeder

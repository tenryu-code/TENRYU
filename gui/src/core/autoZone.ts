export interface AutoZoneRegionTS {
  rEnd: number;
  nz: number;
  rhoRef: number;
  isVoid: boolean;
  materialGroup: string;
}

export interface AutoZoneConfigTS {
  geometryCode: number;
  massRatioMax: number;
  nBridgeMin: number;
  nBridgeMax: number;
  bridgeFracMax: number;
  rhoVoidCut: number;
  drMin: number;
  massRatioHardMax: number;
  maxIter: number;
  bulkMassTol: number;
}

interface RegionState {
  rIn: number;
  rOut: number;
  nz: number;
  rho: number;
  isVoid: boolean;
  materialGroup: string;
  totalMass: number;
  avgDr: number;
}

interface InterfaceCandidate {
  valid: boolean;
  nLeft: number;
  nRight: number;
  q: number;
  mInt: number;
  ifaceMassLeft: number;
  ifaceMassRight: number;
  twoSided: boolean;
  symmetry: number;
  drMargin: number;
}

interface BridgePlan {
  nLeft: number;
  nRight: number;
  active: boolean;
}

interface AutoZoneDiagnosticsTS {
  massRatioMin: number;
  massRatioMax: number;
  massRatioMean: number;
  drMinActual: number;
  nRatioViolations: number;
  warnings: string[];
}

const K_FOUR_PI_OVER_3 = 4.188790204786390984616857844372670512262892532500141;
const K_PI = 3.141592653589793238462643383279502884;
const K_TINY = 1.0e-300;

export function defaultAutoZoneConfig(): AutoZoneConfigTS {
  return {
    geometryCode: 0,
    massRatioMax: 1.3,
    nBridgeMin: 2,
    nBridgeMax: 10,
    bridgeFracMax: 0.25,
    rhoVoidCut: 1.0e-6,
    drMin: 1.0e-8,
    massRatioHardMax: 2.0,
    maxIter: 100,
    bulkMassTol: 1.0e-3,
  };
}

function defaultInterfaceCandidate(): InterfaceCandidate {
  return {
    valid: false,
    nLeft: 0,
    nRight: 0,
    q: 1.0,
    mInt: 0.0,
    ifaceMassLeft: 0.0,
    ifaceMassRight: 0.0,
    twoSided: false,
    symmetry: 0,
    drMargin: 0.0,
  };
}

function shellVolume(r0: number, r1: number, geom: number): number {
  if (geom === 0) {
    return K_FOUR_PI_OVER_3 * (r1 * r1 * r1 - r0 * r0 * r0);
  }
  if (geom === 1) {
    return K_PI * (r1 * r1 - r0 * r0);
  }
  if (geom === 2) {
    return r1 - r0;
  }
  return K_FOUR_PI_OVER_3 * (r1 * r1 * r1 - r0 * r0 * r0);
}

function interfaceMassFloorLeft(rInt: number, rho: number, drMin: number, geom: number): number {
  if (!(rho > 0.0) || !(drMin > 0.0)) {
    return 0.0;
  }
  const rInner = Math.max(0.0, rInt - drMin);
  return rho * shellVolume(rInner, rInt, geom);
}

function interfaceMassFloorRight(rInt: number, rho: number, drMin: number, geom: number): number {
  if (!(rho > 0.0) || !(drMin > 0.0)) {
    return 0.0;
  }
  return rho * shellVolume(rInt, rInt + drMin, geom);
}

function bridgeCap(nz: number, cfg: AutoZoneConfigTS): number {
  const fracCap = Math.floor(cfg.bridgeFracMax * nz);
  return Math.max(0, Math.min(cfg.nBridgeMax, fracCap));
}

function trimBridgeForOverlap(nz: number, cLeft: number[], cRight: number[]): void {
  const minBulk = nz <= 6 ? 1 : 2;
  const maxBridge = Math.max(0, nz - minBulk);
  while (cLeft.length + cRight.length > maxBridge) {
    if (cLeft.length > cRight.length) {
      cLeft.shift();
    } else if (cRight.length > 0) {
      cRight.shift();
    } else if (cLeft.length > 0) {
      cLeft.shift();
    } else {
      break;
    }
  }
}

function buildRegionZoneMasses(
  nz: number,
  mBulk: number,
  cLeft: number[],
  cRight: number[],
): number[] {
  const masses = new Array<number>(nz).fill(mBulk);
  for (let i = 0; i < cLeft.length; ++i) {
    masses[i] = mBulk * cLeft[cLeft.length - 1 - i];
  }
  for (let i = 0; i < cRight.length; ++i) {
    masses[nz - 1 - i] = mBulk * cRight[cRight.length - 1 - i];
  }
  return masses;
}

function betterCandidate(a: InterfaceCandidate, b: InterfaceCandidate): boolean {
  if (!b.valid) {
    return true;
  }
  if (a.twoSided !== b.twoSided) {
    return a.twoSided;
  }
  if (a.symmetry !== b.symmetry) {
    return a.symmetry > b.symmetry;
  }
  if (Math.abs(a.drMargin - b.drMargin) > 1.0e-14) {
    return a.drMargin > b.drMargin;
  }
  return false;
}

function buildRegions(
  rMin: number,
  regions: AutoZoneRegionTS[],
  cfg: AutoZoneConfigTS,
): RegionState[] {
  if (!(rMin >= 0.0) || !Number.isFinite(rMin)) {
    throw new Error("auto-zone requires finite r_min >= 0");
  }
  if (regions.length === 0) {
    throw new Error("auto-zone requires at least one region");
  }

  const out: RegionState[] = [];
  let rIn = rMin;
  for (let i = 0; i < regions.length; ++i) {
    const src = regions[i];
    if (!Number.isFinite(src.rEnd) || !(src.rEnd > rIn)) {
      throw new Error(`auto-zone region[${i}] requires finite r_end > previous boundary`);
    }
    if (src.nz <= 0) {
      throw new Error(`auto-zone region[${i}] requires nz > 0`);
    }
    if (!Number.isFinite(src.rhoRef) || src.rhoRef < 0.0) {
      throw new Error(`auto-zone region[${i}] requires finite rho_ref >= 0`);
    }

    const dst: RegionState = {
      rIn,
      rOut: src.rEnd,
      nz: src.nz,
      rho: src.rhoRef,
      isVoid: src.isVoid || src.rhoRef <= cfg.rhoVoidCut,
      materialGroup: src.materialGroup,
      totalMass: src.rhoRef * shellVolume(rIn, src.rEnd, cfg.geometryCode),
      avgDr: (src.rEnd - rIn) / src.nz,
    };

    if (!dst.isVoid && !(dst.rho > 0.0)) {
      throw new Error("auto-zone non-void region requires rho_ref > 0");
    }

    out.push(dst);
    rIn = src.rEnd;
  }
  return out;
}

function alphaSweep(alphaStart: number, alphaHard: number): number[] {
  const values: number[] = [];
  alphaStart = Math.max(alphaStart, 1.0 + 1.0e-12);
  alphaHard = Math.max(alphaHard, alphaStart);
  values.push(alphaStart);
  if (alphaHard <= alphaStart + 1.0e-14) {
    return values;
  }
  let alpha = alphaStart;
  for (let k = 0; k < 32 && alpha < alphaHard - 1.0e-14; ++k) {
    alpha = Math.min(alphaHard, alpha * 1.08);
    if (alpha > values[values.length - 1] + 1.0e-14) {
      values.push(alpha);
    }
  }
  if (values[values.length - 1] < alphaHard - 1.0e-14) {
    values.push(alphaHard);
  }
  return values;
}

function pushWarningOnce(warnings: string[], warning: string): void {
  if (!warnings.includes(warning)) {
    warnings.push(warning);
  }
}

function computeBaseSideCount(nRequired: number, capThis: number, capOther: number): number {
  let nThis = Math.max(0, Math.min(capThis, Math.floor(nRequired / 2)));
  let nOther = nRequired - nThis;
  if (nOther > capOther) {
    nOther = capOther;
    nThis = Math.min(capThis, nRequired - nOther);
  }
  return nThis;
}

function chooseViolationCandidate(
  massRatio: number,
  mLeft: number,
  mRight: number,
  floorLeft: number,
  floorRight: number,
  capLeft: number,
  capRight: number,
): InterfaceCandidate {
  let best = defaultInterfaceCandidate();
  let bestQ = Number.POSITIVE_INFINITY;

  const capSum = capLeft + capRight;
  for (let nTotal = 0; nTotal <= capSum; ++nTotal) {
    const nLeftMin = Math.max(0, nTotal - capRight);
    const nLeftMax = Math.min(capLeft, nTotal);
    for (let nLeft = nLeftMin; nLeft <= nLeftMax; ++nLeft) {
      const nRight = nTotal - nLeft;

      const q = nTotal > 0 ? Math.pow(massRatio, 1.0 / nTotal) : massRatio;
      const mInt =
        nTotal > 0
          ? Math.pow(mLeft, nRight / nTotal) * Math.pow(mRight, nLeft / nTotal)
          : Math.sqrt(mLeft * mRight);

      const ifaceMassLeft = nLeft > 0 ? mInt : mLeft;
      const ifaceMassRight = nRight > 0 ? mInt : mRight;
      if (ifaceMassLeft + 1.0e-14 < floorLeft || ifaceMassRight + 1.0e-14 < floorRight) {
        continue;
      }

      const marginLeft =
        floorLeft > 0.0 ? ifaceMassLeft / floorLeft : Number.POSITIVE_INFINITY;
      const marginRight =
        floorRight > 0.0 ? ifaceMassRight / floorRight : Number.POSITIVE_INFINITY;
      const cand: InterfaceCandidate = {
        valid: true,
        nLeft,
        nRight,
        q,
        mInt,
        ifaceMassLeft,
        ifaceMassRight,
        twoSided: nLeft > 0 && nRight > 0,
        symmetry: -Math.abs(nLeft - nRight),
        drMargin: Math.min(marginLeft, marginRight),
      };

      if (
        !best.valid ||
        q < bestQ - 1.0e-14 ||
        (Math.abs(q - bestQ) <= 1.0e-14 && betterCandidate(cand, best))
      ) {
        best = cand;
        bestQ = q;
      }
    }
  }
  return best;
}

function coeffSumIncreasing(q: number, n: number): number {
  if (n <= 0) {
    return 0.0;
  }
  let sum = 0.0;
  let qk = 1.0;
  for (let k = 0; k < n; ++k) {
    qk *= q;
    sum += qk;
  }
  return sum;
}

function coeffSumDecreasing(q: number, n: number): number {
  if (n <= 0) {
    return 0.0;
  }
  let sum = 0.0;
  let qk = 1.0;
  for (let k = 0; k < n; ++k) {
    qk /= q;
    sum += qk;
  }
  return sum;
}

function interfaceG(
  q: number,
  nLeft: number,
  nRight: number,
  mLeft: number,
  mRight: number,
  nBulkLeft: number,
  nBulkRight: number,
  otherSumLeft: number,
  otherSumRight: number,
): number {
  const nTotal = nLeft + nRight;
  const sLeft = nBulkLeft + coeffSumIncreasing(q, nLeft) + otherSumLeft;
  const sRight = nBulkRight + coeffSumDecreasing(q, nRight) + otherSumRight;
  const qPow = Math.pow(q, nTotal);
  return qPow * mLeft * sRight - mRight * sLeft;
}

function solveQBisection(
  nLeft: number,
  nRight: number,
  mLeft: number,
  mRight: number,
  nBulkLeft: number,
  nBulkRight: number,
  otherSumLeft: number,
  otherSumRight: number,
): number {
  if (nLeft + nRight <= 0) {
    return 1.0;
  }
  if (!(mLeft > 0.0) || !(mRight > 0.0)) {
    return 1.0;
  }

  const gLo = interfaceG(
    1.0,
    nLeft,
    nRight,
    mLeft,
    mRight,
    nBulkLeft,
    nBulkRight,
    otherSumLeft,
    otherSumRight,
  );
  if (Math.abs(gLo) <= 1.0e-30) {
    return 1.0;
  }
  if (!(gLo < 0.0)) {
    return 1.0;
  }

  let qHi = 2.0;
  let gHi = interfaceG(
    qHi,
    nLeft,
    nRight,
    mLeft,
    mRight,
    nBulkLeft,
    nBulkRight,
    otherSumLeft,
    otherSumRight,
  );
  while (!(gHi > 0.0) && qHi < 1.0e6) {
    qHi = Math.min(1.0e6, qHi * 2.0);
    gHi = interfaceG(
      qHi,
      nLeft,
      nRight,
      mLeft,
      mRight,
      nBulkLeft,
      nBulkRight,
      otherSumLeft,
      otherSumRight,
    );
  }
  if (!(gHi > 0.0)) {
    return qHi;
  }

  let lo = 1.0;
  let hi = qHi;
  for (let iter = 0; iter < 60; ++iter) {
    const mid = 0.5 * (lo + hi);
    const gMid = interfaceG(
      mid,
      nLeft,
      nRight,
      mLeft,
      mRight,
      nBulkLeft,
      nBulkRight,
      otherSumLeft,
      otherSumRight,
    );
    if (gMid > 0.0) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
}

function sumValues(values: number[]): number {
  let sum = 0.0;
  for (const value of values) {
    sum += value;
  }
  return sum;
}

export function computeAutoZoneNodes(
  rMin: number,
  regions: AutoZoneRegionTS[],
  cfg: AutoZoneConfigTS,
): number[] {
  if (!(cfg.massRatioMax > 1.0)) {
    throw new Error("auto-zone requires mass_ratio_max > 1");
  }
  if (!(cfg.massRatioHardMax >= cfg.massRatioMax)) {
    throw new Error("auto-zone requires mass_ratio_hard_max >= mass_ratio_max");
  }
  if (cfg.nBridgeMin < 0 || cfg.nBridgeMax < 0) {
    throw new Error("auto-zone requires n_bridge_min/n_bridge_max >= 0");
  }
  if (!(cfg.bridgeFracMax >= 0.0)) {
    throw new Error("auto-zone requires bridge_frac_max >= 0");
  }
  if (!(cfg.drMin >= 0.0)) {
    throw new Error("auto-zone requires dr_min >= 0");
  }
  if (cfg.maxIter <= 0) {
    throw new Error("auto-zone requires max_iter > 0");
  }
  if (!(cfg.bulkMassTol > 0.0)) {
    throw new Error("auto-zone requires bulk_mass_tol > 0");
  }

  const rs = buildRegions(rMin, regions, cfg);
  const nRegions = rs.length;

  let totalZones = 0;
  for (const region of rs) {
    totalZones += region.nz;
  }
  if (totalZones <= 0) {
    throw new Error("auto-zone total zone count must be positive");
  }

  let mBulk = new Array<number>(nRegions).fill(0.0);
  let mBulkNew = new Array<number>(nRegions).fill(0.0);
  let prevMBulk = new Array<number>(nRegions).fill(0.0);
  for (let i = 0; i < nRegions; ++i) {
    if (!rs[i].isVoid) {
      mBulk[i] = rs[i].totalMass / rs[i].nz;
    }
  }

  const leftCoeff = Array.from({ length: nRegions }, () => [] as number[]);
  const rightCoeff = Array.from({ length: nRegions }, () => [] as number[]);
  const warnings: string[] = [];
  const plans = Array.from({ length: nRegions > 1 ? nRegions - 1 : 0 }, (): BridgePlan => ({
    nLeft: 0,
    nRight: 0,
    active: false,
  }));

  for (let iface = 0; iface + 1 < nRegions; ++iface) {
    const left = rs[iface];
    const right = rs[iface + 1];
    if (left.isVoid || right.isVoid) {
      continue;
    }

    const mLeft = mBulk[iface];
    const mRight = mBulk[iface + 1];
    if (!(mLeft > 0.0) || !(mRight > 0.0)) {
      continue;
    }

    const ratio = Math.max(mLeft, mRight) / Math.max(Math.min(mLeft, mRight), K_TINY);
    if (!(ratio >= 1.0)) {
      continue;
    }

    const rInt = left.rOut;
    const floorLeft = interfaceMassFloorLeft(rInt, left.rho, cfg.drMin, cfg.geometryCode);
    const floorRight = interfaceMassFloorRight(rInt, right.rho, cfg.drMin, cfg.geometryCode);

    const capLeft = bridgeCap(left.nz, cfg);
    const capRight = bridgeCap(right.nz, cfg);

    const bindLeft = left.avgDr <= 1.01 * cfg.drMin;
    const bindRight = right.avgDr <= 1.01 * cfg.drMin;

    let chosen = defaultInterfaceCandidate();
    let usedAlpha = cfg.massRatioMax;
    let relaxed = false;

    const alphas = alphaSweep(cfg.massRatioMax, cfg.massRatioHardMax);
    for (const alpha of alphas) {
      usedAlpha = alpha;

      const nRequiredRatio =
        ratio <= 1.0 + 1.0e-14 ? 0 : Math.ceil(Math.log(ratio) / Math.log(alpha));
      const nRequired =
        ratio <= 1.0 + 1.0e-14 ? 0 : Math.max(cfg.nBridgeMin, nRequiredRatio);

      const baseLeft = computeBaseSideCount(nRequired, capLeft, capRight);
      const baseRight = nRequired - baseLeft;

      let capLeftEff = capLeft;
      let capRightEff = capRight;
      if (bindLeft) {
        capLeftEff = Math.min(capLeftEff, baseLeft);
      }
      if (bindRight) {
        capRightEff = Math.min(capRightEff, baseRight);
      }

      const capSum = capLeftEff + capRightEff;
      if (nRequired > capSum) {
        continue;
      }

      for (let nTotal = nRequired; nTotal <= capSum; ++nTotal) {
        let bestForTotal = defaultInterfaceCandidate();
        const nLeftMin = Math.max(0, nTotal - capRightEff);
        const nLeftMax = Math.min(capLeftEff, nTotal);
        for (let nLeft = nLeftMin; nLeft <= nLeftMax; ++nLeft) {
          const nRight = nTotal - nLeft;
          const q = nTotal > 0 ? Math.pow(ratio, 1.0 / nTotal) : 1.0;
          if (nTotal > 0 && q > alpha * (1.0 + 1.0e-12)) {
            continue;
          }

          const mInt =
            nTotal > 0
              ? Math.pow(mLeft, nRight / nTotal) * Math.pow(mRight, nLeft / nTotal)
              : Math.sqrt(mLeft * mRight);
          const ifaceMassLeft = nLeft > 0 ? mInt : mLeft;
          const ifaceMassRight = nRight > 0 ? mInt : mRight;

          if (ifaceMassLeft + 1.0e-14 < floorLeft || ifaceMassRight + 1.0e-14 < floorRight) {
            continue;
          }

          const marginLeft =
            floorLeft > 0.0 ? ifaceMassLeft / floorLeft : Number.POSITIVE_INFINITY;
          const marginRight =
            floorRight > 0.0 ? ifaceMassRight / floorRight : Number.POSITIVE_INFINITY;
          const cand: InterfaceCandidate = {
            valid: true,
            nLeft,
            nRight,
            q,
            mInt,
            ifaceMassLeft,
            ifaceMassRight,
            twoSided: nLeft > 0 && nRight > 0,
            symmetry: -Math.abs(nLeft - nRight),
            drMargin: Math.min(marginLeft, marginRight),
          };

          if (betterCandidate(cand, bestForTotal)) {
            bestForTotal = cand;
          }
        }

        if (bestForTotal.valid) {
          chosen = bestForTotal;
          relaxed = alpha > cfg.massRatioMax + 1.0e-14;
          break;
        }
      }

      if (chosen.valid) {
        break;
      }
    }

    if (!chosen.valid) {
      const nRequiredHard =
        ratio <= 1.0 + 1.0e-14
          ? 0
          : Math.max(
              cfg.nBridgeMin,
              Math.ceil(Math.log(ratio) / Math.log(cfg.massRatioHardMax)),
            );
      const baseLeft = computeBaseSideCount(nRequiredHard, capLeft, capRight);
      const baseRight = nRequiredHard - baseLeft;

      let capLeftEff = capLeft;
      let capRightEff = capRight;
      if (bindLeft) {
        capLeftEff = Math.min(capLeftEff, baseLeft);
      }
      if (bindRight) {
        capRightEff = Math.min(capRightEff, baseRight);
      }

      chosen = chooseViolationCandidate(
        ratio,
        mLeft,
        mRight,
        floorLeft,
        floorRight,
        capLeftEff,
        capRightEff,
      );
      if (!chosen.valid) {
        throw new Error(`auto-zone infeasible at interface ${iface}: dr_min hard floor cannot be satisfied`);
      }

      pushWarningOnce(
        warnings,
        `auto-zone interface ${iface} accepted mass-ratio violation: q=${chosen.q} exceeds hard_max=${cfg.massRatioHardMax}`,
      );
    }

    if (relaxed) {
      pushWarningOnce(
        warnings,
        `auto-zone interface ${iface} relaxed mass_ratio_max from ${cfg.massRatioMax} to ${usedAlpha}`,
      );
    }
    if (bindLeft || bindRight) {
      pushWarningOnce(
        warnings,
        `auto-zone interface ${iface} dr_min binding (left=${bindLeft ? "on" : "off"}, right=${bindRight ? "on" : "off"})`,
      );
    }

    plans[iface] = { nLeft: chosen.nLeft, nRight: chosen.nRight, active: chosen.valid };
  }

  for (let i = 0; i < nRegions; ++i) {
    let leftCount = 0;
    let rightCount = 0;
    if (i > 0 && plans[i - 1].active) {
      leftCount = plans[i - 1].nRight;
    }
    if (i + 1 < nRegions && plans[i].active) {
      rightCount = plans[i].nLeft;
    }

    const cLeft = new Array<number>(leftCount).fill(1.0);
    const cRight = new Array<number>(rightCount).fill(1.0);
    trimBridgeForOverlap(rs[i].nz, cLeft, cRight);
    leftCount = cLeft.length;
    rightCount = cRight.length;

    if (i > 0 && plans[i - 1].active) {
      plans[i - 1].nRight = leftCount;
    }
    if (i + 1 < nRegions && plans[i].active) {
      plans[i].nLeft = rightCount;
    }
  }

  const otherSumLeft = new Array<number>(nRegions).fill(0.0);
  const otherSumRight = new Array<number>(nRegions).fill(0.0);

  for (let couplingPass = 0; couplingPass < 3; ++couplingPass) {
    prevMBulk = [...mBulk];
    mBulkNew = [...prevMBulk];

    for (let iface = 0; iface + 1 < nRegions; ++iface) {
      if (!plans[iface].active) {
        rightCoeff[iface].length = 0;
        leftCoeff[iface + 1].length = 0;
        otherSumRight[iface] = 0.0;
        otherSumLeft[iface + 1] = 0.0;
        continue;
      }
      if (rs[iface].isVoid || rs[iface + 1].isVoid) {
        rightCoeff[iface].length = 0;
        leftCoeff[iface + 1].length = 0;
        otherSumRight[iface] = 0.0;
        otherSumLeft[iface + 1] = 0.0;
        continue;
      }

      let nLeft = plans[iface].nLeft;
      let nRight = plans[iface].nRight;

      const capLeft = bridgeCap(rs[iface].nz, cfg);
      const capRight = bridgeCap(rs[iface + 1].nz, cfg);
      const rInt = rs[iface].rOut;
      const floorLeftVal = interfaceMassFloorLeft(
        rInt,
        rs[iface].rho,
        cfg.drMin,
        cfg.geometryCode,
      );
      const floorRightVal = interfaceMassFloorRight(
        rInt,
        rs[iface + 1].rho,
        cfg.drMin,
        cfg.geometryCode,
      );

      let q = 1.0;

      for (let adjIter = 0; adjIter < 20; ++adjIter) {
        const leftBridgeOfLeft = iface > 0 && plans[iface - 1].active ? plans[iface - 1].nRight : 0;
        const rightBridgeOfRight =
          iface + 2 < nRegions && plans[iface + 1].active ? plans[iface + 1].nLeft : 0;
        const nBulkLeft = rs[iface].nz - leftBridgeOfLeft - nLeft;
        const nBulkRight = rs[iface + 1].nz - nRight - rightBridgeOfRight;
        if (nBulkLeft < 2 || nBulkRight < 2) break;

        q = solveQBisection(
          nLeft,
          nRight,
          rs[iface].totalMass,
          rs[iface + 1].totalMass,
          nBulkLeft,
          nBulkRight,
          otherSumLeft[iface],
          otherSumRight[iface + 1],
        );

        const sLVal = nBulkLeft + coeffSumIncreasing(q, nLeft) + otherSumLeft[iface];
        const sRVal = nBulkRight + coeffSumDecreasing(q, nRight) + otherSumRight[iface + 1];
        const mIfaceL =
          nLeft > 0 && sLVal > 0.0
            ? (rs[iface].totalMass / sLVal) * Math.pow(q, nLeft)
            : 0.0;
        const mIfaceR =
          nRight > 0 && sRVal > 0.0
            ? (rs[iface + 1].totalMass / sRVal) * Math.pow(q, -nRight)
            : 0.0;

        let reduced = false;
        if (nRight > 0 && mIfaceR + 1.0e-14 < floorRightVal) {
          --nRight;
          reduced = true;
        }
        if (nLeft > 0 && mIfaceL + 1.0e-14 < floorLeftVal) {
          --nLeft;
          reduced = true;
        }
        if (reduced) continue;

        if (q > cfg.massRatioMax * (1.0 + 1.0e-12)) {
          const effCapLeft = Math.min(capLeft, rs[iface].nz - leftBridgeOfLeft - 2);
          const effCapRight = Math.min(capRight, rs[iface + 1].nz - rightBridgeOfRight - 2);
          let added = false;
          if (nLeft < effCapLeft && nLeft <= nRight) {
            ++nLeft;
            added = true;
          } else if (nRight < effCapRight) {
            ++nRight;
            added = true;
          } else if (nLeft < effCapLeft) {
            ++nLeft;
            added = true;
          }
          if (added) continue;
        }

        break;
      }

      plans[iface].nLeft = nLeft;
      plans[iface].nRight = nRight;

      const leftBridgeOfLeftFinal =
        iface > 0 && plans[iface - 1].active ? plans[iface - 1].nRight : 0;
      const rightBridgeOfRightFinal =
        iface + 2 < nRegions && plans[iface + 1].active ? plans[iface + 1].nLeft : 0;
      const nBulkLeftFinal = rs[iface].nz - leftBridgeOfLeftFinal - nLeft;
      const nBulkRightFinal = rs[iface + 1].nz - nRight - rightBridgeOfRightFinal;

      rightCoeff[iface].length = 0;
      for (let k = 0; k < nLeft; ++k) {
        rightCoeff[iface].push(Math.pow(q, k + 1));
      }

      leftCoeff[iface + 1].length = 0;
      for (let k = 0; k < nRight; ++k) {
        leftCoeff[iface + 1].push(Math.pow(q, -(k + 1)));
      }

      otherSumRight[iface] = sumValues(rightCoeff[iface]);
      otherSumLeft[iface + 1] = sumValues(leftCoeff[iface + 1]);

      const sLeft = nBulkLeftFinal + otherSumLeft[iface] + otherSumRight[iface];
      const sRight = nBulkRightFinal + otherSumLeft[iface + 1] + otherSumRight[iface + 1];
      if (sLeft > 0.0) {
        mBulkNew[iface] = rs[iface].totalMass / sLeft;
      }
      if (sRight > 0.0) {
        mBulkNew[iface + 1] = rs[iface + 1].totalMass / sRight;
      }
    }

    for (let i = 0; i < nRegions; ++i) {
      if (rs[i].isVoid) {
        mBulkNew[i] = 0.0;
      }
    }
    const oldMBulk = mBulk;
    mBulk = mBulkNew;
    mBulkNew = oldMBulk;
  }

  for (let i = 0; i < nRegions; ++i) {
    if (rs[i].isVoid) {
      mBulk[i] = 0.0;
      continue;
    }
    const nBulk = rs[i].nz - leftCoeff[i].length - rightCoeff[i].length;
    if (nBulk <= 0) {
      continue;
    }
    const sumLeft = sumValues(leftCoeff[i]);
    const sumRight = sumValues(rightCoeff[i]);
    const denom = nBulk + sumLeft + sumRight;
    if (denom > 0.0) {
      mBulk[i] = rs[i].totalMass / denom;
    }
  }

  const nodes: number[] = [rMin];
  const zoneMasses: number[] = [];
  const zoneNonvoid: boolean[] = [];

  for (let i = 0; i < nRegions; ++i) {
    const region = rs[i];
    const rIn = nodes[nodes.length - 1];
    if (region.isVoid) {
      for (let k = 0; k < region.nz; ++k) {
        const u = (k + 1) / region.nz;
        let rNext = rIn + (region.rOut - rIn) * u;
        if (k + 1 === region.nz) {
          rNext = region.rOut;
        }
        const mZone = region.rho * shellVolume(nodes[nodes.length - 1], rNext, cfg.geometryCode);
        zoneMasses.push(mZone);
        zoneNonvoid.push(false);
        nodes.push(rNext);
      }
      continue;
    }

    const mZone = buildRegionZoneMasses(
      region.nz,
      mBulk[i],
      leftCoeff[i],
      rightCoeff[i],
    );
    let rCubed = rIn * rIn * rIn;
    let rSq = rIn * rIn;
    let xLin = rIn;
    for (let k = 0; k < region.nz; ++k) {
      const mass = mZone[k];
      let rNext: number;
      if (cfg.geometryCode === 1) {
        rSq += mass / (Math.max(region.rho, K_TINY) * K_PI);
        rNext = Math.sqrt(Math.max(rSq, 0.0));
      } else if (cfg.geometryCode === 2) {
        xLin += mass / Math.max(region.rho, K_TINY);
        rNext = xLin;
      } else {
        const delta = mass / (Math.max(region.rho, K_TINY) * K_FOUR_PI_OVER_3);
        rCubed += delta;
        rNext = Math.cbrt(Math.max(rCubed, 0.0));
      }
      if (k + 1 === region.nz) {
        rNext = region.rOut;
      }
      zoneMasses.push(mass);
      zoneNonvoid.push(true);
      nodes.push(rNext);
    }
  }

  if (nodes.length !== totalZones + 1) {
    throw new Error("auto-zone internal error: node count mismatch");
  }

  for (let i = 0; i < totalZones; ++i) {
    if (!(nodes[i + 1] > nodes[i])) {
      throw new Error("auto-zone produced non-monotonic nodes");
    }
  }

  let ifaceZone = 0;
  for (let i = 0; i + 1 < nRegions; ++i) {
    ifaceZone += rs[i].nz;
    if (rs[i].isVoid || rs[i + 1].isVoid) {
      continue;
    }
    const drLeft = nodes[ifaceZone] - nodes[ifaceZone - 1];
    const drRight = nodes[ifaceZone + 1] - nodes[ifaceZone];
    if (drLeft + 1.0e-14 < cfg.drMin || drRight + 1.0e-14 < cfg.drMin) {
      throw new Error(
        `auto-zone dr_min hard constraint violated at interface ${i} (dr_left=${drLeft}, dr_right=${drRight}, dr_min=${cfg.drMin})`,
      );
    }
  }

  let drMinActual = Number.POSITIVE_INFINITY;
  for (let i = 0; i < totalZones; ++i) {
    drMinActual = Math.min(drMinActual, nodes[i + 1] - nodes[i]);
  }

  let ratioMin = Number.POSITIVE_INFINITY;
  let ratioMax = 1.0;
  let ratioSum = 0.0;
  let ratioCount = 0;
  let nViol = 0;
  for (let i = 0; i + 1 < totalZones; ++i) {
    if (!zoneNonvoid[i] || !zoneNonvoid[i + 1]) {
      continue;
    }
    const m0 = zoneMasses[i];
    const m1 = zoneMasses[i + 1];
    if (!(m0 > 0.0) || !(m1 > 0.0)) {
      continue;
    }
    const ratio = Math.max(m0, m1) / Math.min(m0, m1);
    ratioMin = Math.min(ratioMin, ratio);
    ratioMax = Math.max(ratioMax, ratio);
    ratioSum += ratio;
    ++ratioCount;
    if (ratio > cfg.massRatioMax * (1.0 + 1.0e-12)) {
      ++nViol;
    }
  }

  const localDiag: AutoZoneDiagnosticsTS = {
    massRatioMin: ratioCount > 0 ? ratioMin : 1.0,
    massRatioMax: ratioCount > 0 ? ratioMax : 1.0,
    massRatioMean: ratioCount > 0 ? ratioSum / ratioCount : 1.0,
    drMinActual: Number.isFinite(drMinActual) ? drMinActual : 0.0,
    nRatioViolations: nViol,
    warnings,
  };
  void localDiag;

  return nodes;
}

import { t } from "../../i18n";
import {
  driveMode,
  effectiveLaserRaysPerBeam,
  makeHotEChannel,
  setDriveMode,
  type DriveMode,
} from "../../core/deck/formState";
import { toCanonical } from "../../core/units";
import { useApp } from "../../store";
import { NumInput, QInput, SelectField, SwitchField, TextField } from "../fields";
import { Button } from "@tenryu-common/ui/kit";
import WaveformEditor from "../WaveformEditor";
import { BEAM_PRESETS, expandedPairCount, PAIR_CAP } from "../../core/deck/beamPresets";

export default function LaserSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const rayOutputEffectiveLimit = effectiveLaserRaysPerBeam(form);
  const cbetPreset = BEAM_PRESETS[form.laser.cbet.portPreset];
  const cbetPairCount = expandedPairCount(
    cbetPreset.ports.length,
    form.laser.cbet.nImpactBins,
  );
  const is1d = form.main.dimension !== "2D_RZ";
  const collapsedBeamEditor = is1d || form.laser.cbet.enabled;
  const displayedBeams = collapsedBeamEditor
    ? form.laser.beams.slice(0, 1)
    : form.laser.beams;
  return (
    <div className="max-w-xl flex flex-col gap-1">
      <h1 className="mb-2 text-base font-semibold">{m.form.driveSectionTitle}</h1>
        <>
          <SelectField
            label={m.form.driveMode}
            value={driveMode(form)}
            options={[
              { value: "none", label: m.form.driveModeNone },
              { value: "laser", label: m.form.driveModeLaser },
              { value: "radiation_tr", label: m.form.driveModeTr },
              ...(form.main.dimension === "1D_SPH"
                ? [{ value: "pressure", label: m.form.driveModePressure }]
                : []),
            ]}
            onChange={(v) => update((f) => { setDriveMode(f, v as DriveMode); })}
          />
          {form.laser.enabled && (
            <>
              <SelectField
                label={m.form.wavelengthPreset}
                value={
                  ["1053", "526.5", "351"].includes(String(form.laser.wavelengthNm))
                    ? String(form.laser.wavelengthNm)
                    : "custom"
                }
                options={[
                  { value: "1053", label: m.form.wl1w },
                  { value: "526.5", label: m.form.wl2w },
                  { value: "351", label: m.form.wl3w },
                  { value: "custom", label: m.form.wlCustom },
                ]}
                onChange={(v) => {
                  if (v !== "custom") update((f) => { f.laser.wavelengthNm = Number(v); });
                }}
              />
              <NumInput
                label={m.form.wavelength}
                value={form.laser.wavelengthNm}
                onChange={(n) => update((f) => { f.laser.wavelengthNm = n ?? 0; })}
              />
              {form.main.dimension === "2D_RZ" ? (
                <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.laser2dModeNote}</p>
              ) : (
                <SelectField
                  label={m.form.laserMode}
                  value={form.laser.mode}
                  options={[
                    { value: "radial_absorption_1d", label: m.form.laserModeRadial },
                    { value: "raytrace_2d", label: m.form.laserModeRaytrace },
                  ]}
                  onChange={(v) => update((f) => { f.laser.mode = v as never; })}
                />
              )}
              {form.laser.mode === "raytrace_2d" && (
                <NumInput
                  int
                  label={m.form.raysPerBeam}
                  value={form.laser.raysPerBeam}
                  onChange={(n) => update((f) => { f.laser.raysPerBeam = n ?? 1000; })}
                />
              )}
              <SwitchField
                label={m.form.rayOutputTrajectory}
                checked={form.laser.rayOutputTrajectory}
                onChange={(b) => update((f) => { f.laser.rayOutputTrajectory = b; })}
              />
              {form.laser.rayOutputTrajectory && (
                <div>
                  <NumInput
                    int
                    label={m.form.rayOutputCount}
                    value={form.laser.rayOutputCount}
                    onChange={(n) => update((f) => { f.laser.rayOutputCount = n ?? 200; })}
                  />
                  <p className="mb-2 text-xs" style={{ color: "var(--fg-secondary)" }}>
                    {m.form.rayOutputEffectiveLimit(rayOutputEffectiveLimit)}
                  </p>
                </div>
              )}
              <SelectField
                label={m.form.waveformMode}
                value={form.laser.waveformMode}
                options={[
                  { value: "square", label: m.form.waveformSquare },
                  { value: "gaussian", label: m.form.waveformGaussian },
                  { value: "table", label: m.form.waveformTable },
                ]}
                onChange={(v) => update((f) => { f.laser.waveformMode = v as never; })}
              />
              {form.laser.waveformMode === "square" ? (
                <>
                  <QInput
                    label={m.form.power}
                    kind="power"
                    value={form.laser.powerW}
                    onChange={(q) => update((f) => { f.laser.powerW = q; })}
                  />
                  <QInput
                    label={m.form.pulseDuration}
                    kind="time"
                    value={form.laser.pulseDuration}
                    onChange={(q) => update((f) => { f.laser.pulseDuration = q; })}
                  />
                  <QInput
                    label={m.form.riseTime}
                    kind="time"
                    value={form.laser.riseTime}
                    onChange={(q) => update((f) => { f.laser.riseTime = q; })}
                  />
                  <QInput
                    label={m.form.fallTime}
                    kind="time"
                    value={form.laser.fallTime}
                    onChange={(q) => update((f) => { f.laser.fallTime = q; })}
                  />
                </>
              ) : form.laser.waveformMode === "gaussian" ? (
                <>
                  <SelectField
                    label={m.form.gaussianSpec}
                    value={form.laser.gaussianSpec}
                    options={[
                      { value: "peak", label: m.form.gaussianSpecPeak },
                      { value: "energy", label: m.form.gaussianSpecEnergy },
                    ]}
                    onChange={(v) => update((f) => { f.laser.gaussianSpec = v as never; })}
                  />
                  {form.laser.gaussianSpec === "peak" ? (
                    <QInput
                      label={m.form.gaussianPeak}
                      kind="power"
                      value={form.laser.gaussianPeakW}
                      onChange={(q) => update((f) => { f.laser.gaussianPeakW = q; })}
                    />
                  ) : (
                    <NumInput
                      label={m.form.gaussianEnergyJ}
                      value={form.laser.gaussianEnergyJ}
                      onChange={(n) => update((f) => { f.laser.gaussianEnergyJ = n ?? 0; })}
                    />
                  )}
                  <QInput
                    label={m.form.gaussianFwhm}
                    kind="time"
                    value={form.laser.gaussianFwhm}
                    onChange={(q) => update((f) => { f.laser.gaussianFwhm = q; })}
                  />
                  <QInput
                    label={m.form.gaussianCenter}
                    kind="time"
                    value={form.laser.gaussianCenter}
                    onChange={(q) => update((f) => { f.laser.gaussianCenter = q; })}
                  />
                  {toCanonical(form.laser.gaussianFwhm, "time") > 0 && (
                    <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
                      {form.laser.gaussianSpec === "peak"
                        ? `${m.form.gaussianDerivedE}: ${(
                            (toCanonical(form.laser.gaussianPeakW, "power") *
                              toCanonical(form.laser.gaussianFwhm, "time")) /
                            0.9394372786996513
                          ).toPrecision(4)} J`
                        : `${m.form.gaussianDerivedP0}: ${(
                            (form.laser.gaussianEnergyJ * 0.9394372786996513) /
                            toCanonical(form.laser.gaussianFwhm, "time") /
                            1e12
                          ).toPrecision(4)} TW`}
                    </p>
                  )}
                </>
              ) : (
                <WaveformEditor
                  points={form.laser.waveformPoints}
                  onChange={(pts) => update((f) => { f.laser.waveformPoints = pts; })}
                  yLabel={m.form.laserPointV}
                  importHint={m.form.wfImportHint}
                />
              )}
              <h2 className="mt-3 text-sm font-semibold">
                {collapsedBeamEditor
                  ? form.laser.cbet.enabled
                    ? m.form.beamsRefTitle
                    : m.form.beamsRefTitle1d
                  : m.form.beamsTitle}
              </h2>
              {is1d && form.laser.beams.length > 1 && (
                <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
                  {m.form.beamsRefExtraIgnored}
                </p>
              )}
              {displayedBeams.map((b, i) => (
                <div key={i} className="mb-2 rounded border p-2" style={{ borderColor: "var(--separator)" }}>
                  {!collapsedBeamEditor && (
                    <>
                      <TextField
                        label={m.form.beamName}
                        value={b.name}
                        onChange={(v) => update((f) => { f.laser.beams[i].name = v; })}
                      />
                      <NumInput
                        label={m.form.beamFraction}
                        value={b.powerFraction}
                        onChange={(n) => update((f) => { f.laser.beams[i].powerFraction = n ?? 1; })}
                      />
                    </>
                  )}
                  <NumInput
                    label={m.form.fNumber}
                    value={b.fNumber}
                    onChange={(n) => update((f) => { f.laser.beams[i].fNumber = n ?? 6.7; })}
                  />
                  <NumInput
                    label={m.form.focusZ}
                    value={b.focusZUm}
                    onChange={(n) => update((f) => { f.laser.beams[i].focusZUm = n ?? 0; })}
                  />
                  {!collapsedBeamEditor && (
                    form.main.dimension !== "2D_RZ" ? (
                      <SelectField
                        label={m.form.beamAxialDir}
                        value={b.axialDirection}
                        options={[
                          { value: "minus_z", label: m.form.beamDirMinusZ },
                          { value: "plus_z", label: m.form.beamDirPlusZ },
                        ]}
                        onChange={(v) => update((f) => { f.laser.beams[i].axialDirection = v as never; })}
                      />
                    ) : (
                      <>
                        <NumInput
                          label={m.form.beamDirX}
                          value={b.dirX}
                          onChange={(n) => update((f) => { f.laser.beams[i].dirX = n ?? 0; })}
                        />
                        <NumInput
                          label={m.form.beamDirY}
                          value={b.dirY}
                          onChange={(n) => update((f) => { f.laser.beams[i].dirY = n ?? 0; })}
                        />
                        <NumInput
                          label={m.form.beamDirZ}
                          value={b.dirZ}
                          onChange={(n) => update((f) => { f.laser.beams[i].dirZ = n ?? 0; })}
                        />
                      </>
                    )
                  )}
                  <SelectField
                    label={m.form.beamProfileModel}
                    value={b.profileModel}
                    options={[
                      { value: "super_gaussian", label: m.form.beamProfileSg },
                      { value: "table", label: m.form.beamProfileTable },
                    ]}
                    onChange={(v) => update((f) => { f.laser.beams[i].profileModel = v as never; })}
                  />
                  {b.profileModel === "super_gaussian" ? (
                    <>
                      <NumInput
                        label={m.form.w0}
                        value={b.w0Um}
                        onChange={(n) => update((f) => { f.laser.beams[i].w0Um = n ?? 0; })}
                      />
                      <NumInput
                        int
                        label={m.form.sgM}
                        value={b.superGaussianM}
                        onChange={(n) => update((f) => { f.laser.beams[i].superGaussianM = n ?? 1; })}
                      />
                    </>
                  ) : (
                    <WaveformEditor
                      points={form.laser.beams[i].profilePoints}
                      onChange={(pts) => update((f) => { f.laser.beams[i].profilePoints = pts; })}
                      yLabel={m.form.beamProfileAxis}
                    />
                  )}
                  {!collapsedBeamEditor && (
                    <Button
                      variant="danger"
                      disabled={form.laser.beams.length === 1}
                      onClick={() => update((f) => { f.laser.beams.splice(i, 1); })}
                    >
                      削除
                    </Button>
                  )}
                </div>
              ))}
              {!collapsedBeamEditor && (
                <Button
                  onClick={() => update((f) => {
                    const index = f.laser.beams.length;
                    f.laser.beams.push({
                      ...f.laser.beams[index - 1],
                      name: `beam_${String(index).padStart(2, "0")}`,
                    });
                  })}
                >
                  {m.form.beamAdd}
                </Button>
              )}
              <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.laser1dNote}</p>
              {form.main.dimension !== "2D_RZ" && (
                <>
                  <h2 className="mt-3 text-sm font-semibold">{m.form.cbetTitle}</h2>
                  <SwitchField
                    label={m.form.cbetEnable}
                    checked={form.laser.cbet.enabled}
                    onChange={(b) => update((f) => {
                      f.laser.cbet.enabled = b;
                      if (b) f.laser.mode = "raytrace_2d";
                    })}
                  />
                  <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
                    {m.form.cbetNeedsRaytraceHint}
                  </p>
                  {form.laser.cbet.enabled && (
                    <>
                      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
                        {m.form.cbetGeometryFixed}
                      </p>
                      <SelectField
                        label={m.form.cbetPortPreset}
                        value={form.laser.cbet.portPreset}
                        options={[
                          { value: "gxii", label: m.form.cbetPortPresetGxii },
                          { value: "omega", label: m.form.cbetPortPresetOmega },
                          { value: "nif", label: m.form.cbetPortPresetNif },
                        ]}
                        onChange={(v) => update((f) => {
                          const preset = BEAM_PRESETS[v as keyof typeof BEAM_PRESETS];
                          f.laser.cbet.portPreset = preset.id;
                          f.laser.cbet.nImpactBins = preset.recommendedBins;
                        })}
                      />
                      <p
                        className="text-xs"
                        style={{
                          color: cbetPairCount > PAIR_CAP ? "var(--err)" : "var(--fg-secondary)",
                        }}
                      >
                        {m.form.cbetPortInfo(
                          cbetPreset.ports.length,
                          cbetPairCount,
                          PAIR_CAP,
                        )}
                      </p>
                      <NumInput
                        int
                        label={m.form.cbetNSectionPhi}
                        value={form.laser.cbet.nSectionPhi}
                        onChange={(n) => update((f) => { f.laser.cbet.nSectionPhi = n ?? 8; })}
                      />
                      <NumInput
                        label={m.form.cbetDetuneSplit}
                        value={form.laser.cbet.detuneSplitNm}
                        hint={m.form.cbetDetuneHint}
                        onChange={(n) => update((f) => { f.laser.cbet.detuneSplitNm = n ?? 0; })}
                      />
                      <NumInput
                        label={m.form.cbetFCbet}
                        value={form.laser.cbet.fCbet}
                        onChange={(n) => update((f) => { f.laser.cbet.fCbet = n ?? 1; })}
                      />
                      <details>
                        <summary>{m.form.cbetAdvanced}</summary>
                        <NumInput
                          label={m.form.cbetAlphaIaw}
                          value={form.laser.cbet.alphaIaw}
                          onChange={(n) => update((f) => { f.laser.cbet.alphaIaw = n ?? 0.2; })}
                        />
                        <NumInput
                          label={m.form.cbetThetaCap}
                          value={form.laser.cbet.thetaCap}
                          onChange={(n) => update((f) => { f.laser.cbet.thetaCap = n ?? 0.3; })}
                        />
                        <NumInput
                          label={m.form.cbetTol}
                          value={form.laser.cbet.tol}
                          onChange={(n) => update((f) => { f.laser.cbet.tol = n ?? 1e-3; })}
                        />
                        <NumInput
                          int
                          label={m.form.cbetMaxIters}
                          value={form.laser.cbet.maxIters}
                          onChange={(n) => update((f) => { f.laser.cbet.maxIters = n ?? 50; })}
                        />
                        <NumInput
                          int
                          label={m.form.cbetBins}
                          value={form.laser.cbet.nImpactBins}
                          onChange={(n) => update((f) => { f.laser.cbet.nImpactBins = n ?? 4; })}
                        />
                        <NumInput
                          label={m.form.cbetNeCutoff}
                          value={form.laser.cbet.neFracCutoff}
                          onChange={(n) => update((f) => { f.laser.cbet.neFracCutoff = n ?? 0.95; })}
                        />
                        <NumInput
                          label={m.form.cbetKaFloor}
                          value={form.laser.cbet.kAFloor}
                          onChange={(n) => update((f) => { f.laser.cbet.kAFloor = n ?? 1e-6; })}
                        />
                      </details>
                    </>
                  )}
                </>
              )}
              {form.main.dimension !== "2D_RZ" && (
                <h2 className="mt-3 text-sm font-semibold">{m.form.hoteTitle}</h2>
              )}
              {form.main.dimension !== "2D_RZ" && (
                <SwitchField
                  label={m.form.hoteEnable}
                  checked={form.laser.hotE.enabled}
                  onChange={(b) => update((f) => { f.laser.hotE.enabled = b; })}
                />
              )}
              {form.main.dimension !== "2D_RZ" && form.laser.hotE.enabled && (
                <>
                  <SelectField
                    label={m.form.hoteSourceLayout}
                    value={form.laser.hotE.useChannels ? "channels" : "single"}
                    options={[
                      { value: "single", label: m.form.hoteLayoutSingle },
                      { value: "channels", label: m.form.hoteLayoutChannels },
                    ]}
                    onChange={(v) => update((f) => {
                      f.laser.hotE.useChannels = v === "channels";
                      if (f.laser.hotE.useChannels && f.laser.hotE.channels.length === 0) {
                        f.laser.hotE.channels.push(makeHotEChannel("tpd"));
                      }
                    })}
                  />
                  {!form.laser.hotE.useChannels ? (
                    <>
                      <NumInput
                        label={m.form.hoteNcFraction}
                        value={form.laser.hotE.sourceNcFraction}
                        onChange={(n) => update((f) => { f.laser.hotE.sourceNcFraction = n ?? 0.25; })}
                      />
                      <SelectField
                        label={m.form.hoteEtaMode}
                        value={form.laser.hotE.etaMode}
                        options={[
                          { value: "constant", label: m.form.hoteEtaConstant },
                          { value: "table", label: m.form.hoteEtaTable },
                        ]}
                        onChange={(v) => update((f) => { f.laser.hotE.etaMode = v as never; })}
                      />
                      {form.laser.hotE.etaMode === "constant" ? (
                        <NumInput
                          label={m.form.hoteEta}
                          value={form.laser.hotE.etaHot}
                          onChange={(n) => update((f) => { f.laser.hotE.etaHot = n ?? 0; })}
                        />
                      ) : (
                        <WaveformEditor
                          points={form.laser.hotE.etaPoints}
                          onChange={(pts) => update((f) => { f.laser.hotE.etaPoints = pts; })}
                          yLabel={m.form.hoteEtaAxis}
                        />
                      )}
                      <NumInput
                        label={m.form.hoteTHot}
                        value={form.laser.hotE.THotEV}
                        onChange={(n) => update((f) => { f.laser.hotE.THotEV = n ?? 5.0e4; })}
                      />
                      <SelectField
                        label={m.form.hoteAngular}
                        value={form.laser.hotE.angularModel}
                        options={[
                          { value: "cone", label: m.form.hoteAngularCone },
                          { value: "radial", label: m.form.hoteAngularRadial },
                        ]}
                        onChange={(v) => update((f) => { f.laser.hotE.angularModel = v as never; })}
                      />
                      {form.laser.hotE.angularModel === "cone" ? (
                        <NumInput
                          label={m.form.hoteTheta}
                          value={form.laser.hotE.thetaDivDeg}
                          onChange={(n) => update((f) => { f.laser.hotE.thetaDivDeg = n ?? 60; })}
                        />
                      ) : (
                        <SelectField
                          label={m.form.hoteInnerBc}
                          value={form.laser.hotE.innerBc}
                          options={[
                            { value: "deposit_residual", label: m.form.hoteInnerBcDeposit },
                            { value: "escape", label: m.form.hoteInnerBcEscape },
                          ]}
                          onChange={(v) => update((f) => { f.laser.hotE.innerBc = v as never; })}
                        />
                      )}
                      <SwitchField
                        label={m.form.hoteSubtract}
                        checked={form.laser.hotE.subtractFromLaser}
                        onChange={(b) => update((f) => { f.laser.hotE.subtractFromLaser = b; })}
                      />
                      <details>
                        <summary>{m.form.hoteAdvanced}</summary>
                        <NumInput
                          int
                          label={m.form.hoteGroups}
                          value={form.laser.hotE.nEnergyGroups}
                          onChange={(n) => update((f) => { f.laser.hotE.nEnergyGroups = n ?? 30; })}
                        />
                        <NumInput
                          label={m.form.hoteEMin}
                          value={form.laser.hotE.EMinOverTh}
                          onChange={(n) => update((f) => { f.laser.hotE.EMinOverTh = n ?? 0.2; })}
                        />
                        <NumInput
                          label={m.form.hoteEMax}
                          value={form.laser.hotE.EMaxOverTh}
                          onChange={(n) => update((f) => { f.laser.hotE.EMaxOverTh = n ?? 8; })}
                        />
                        <NumInput
                          int
                          label={m.form.hoteNMu}
                          value={form.laser.hotE.nMu}
                          onChange={(n) => update((f) => { f.laser.hotE.nMu = n ?? 6; })}
                        />
                        <NumInput
                          int
                          label={m.form.hoteNPhi}
                          value={form.laser.hotE.nPhi}
                          onChange={(n) => update((f) => { f.laser.hotE.nPhi = n ?? 8; })}
                        />
                        <NumInput
                          label={m.form.hoteSourceLimit}
                          value={form.laser.hotE.explicitSourceLimit}
                          onChange={(n) => update((f) => { f.laser.hotE.explicitSourceLimit = n ?? 0.2; })}
                        />
                      </details>
                    </>
                  ) : (
                    <>
                      <SelectField
                        label={m.form.hoteEtaEvolution}
                        value={form.laser.hotE.etaEvolution}
                        options={[
                          { value: "legacy", label: m.form.hoteEtaLegacy },
                          { value: "model", label: m.form.hoteEtaModel },
                        ]}
                        onChange={(v) => update((f) => {
                          f.laser.hotE.etaEvolution = v as "legacy" | "model";
                          if (v === "model") f.laser.hotE.subtractFromLaser = true;
                        })}
                      />
                      {form.laser.hotE.etaEvolution === "model" && (
                        <>
                          <NumInput
                            label={m.form.hoteLnFilterTau}
                            value={form.laser.hotE.lnFilterTauS}
                            onChange={(n) => update((f) => { f.laser.hotE.lnFilterTauS = n ?? 5.0e-12; })}
                            hint={m.form.hoteLnFilterTauHint}
                          />
                          <NumInput
                            label={m.form.hoteEtaTotalCap}
                            value={form.laser.hotE.etaTotalCap}
                            onChange={(n) => update((f) => { f.laser.hotE.etaTotalCap = n ?? 0.08; })}
                            hint={m.form.hoteEtaTotalCapHint}
                          />
                        </>
                      )}
                      <SelectField
                        label={m.form.hoteAngular}
                        value={form.laser.hotE.angularModel}
                        options={[
                          { value: "cone", label: m.form.hoteAngularCone },
                          { value: "radial", label: m.form.hoteAngularRadial },
                        ]}
                        onChange={(v) => update((f) => { f.laser.hotE.angularModel = v as never; })}
                      />
                      {form.laser.hotE.angularModel === "radial" && (
                        <SelectField
                          label={m.form.hoteInnerBc}
                          value={form.laser.hotE.innerBc}
                          options={[
                            { value: "deposit_residual", label: m.form.hoteInnerBcDeposit },
                            { value: "escape", label: m.form.hoteInnerBcEscape },
                          ]}
                          onChange={(v) => update((f) => { f.laser.hotE.innerBc = v as never; })}
                        />
                      )}
                      <SwitchField
                        label={m.form.hoteSubtract}
                        checked={form.laser.hotE.subtractFromLaser}
                        onChange={(b) => update((f) => { f.laser.hotE.subtractFromLaser = b; })}
                        disabled={form.laser.hotE.etaEvolution === "model"}
                        hint={
                          form.laser.hotE.etaEvolution === "model"
                            ? m.form.hoteSubtractForcedHint
                            : undefined
                        }
                      />
                      <details>
                        <summary>{m.form.hoteAdvanced}</summary>
                        <NumInput
                          label={m.form.hoteSourceLimit}
                          value={form.laser.hotE.explicitSourceLimit}
                          onChange={(n) => update((f) => { f.laser.hotE.explicitSourceLimit = n ?? 0.2; })}
                        />
                      </details>
                      {form.laser.hotE.channels.map((ch, i) => (
                        <div
                          key={i}
                          className="mb-2 rounded border p-2"
                          style={{ borderColor: "var(--separator)" }}
                        >
                          <div className="mb-1 flex items-center justify-between">
                            <h3 className="text-sm font-semibold">
                              {m.form.hoteChannel} {i + 1}
                            </h3>
                            <Button
                              variant="danger"
                              onClick={() => update((f) => { f.laser.hotE.channels.splice(i, 1); })}
                            >
                              {m.form.hoteChannelRemove}
                            </Button>
                          </div>
                          <SelectField
                            label="mechanism"
                            value={ch.mechanism}
                            options={[
                              ...(form.laser.hotE.etaEvolution === "model"
                                ? []
                                : [{ value: "cone", label: "cone" }]),
                              { value: "tpd", label: "tpd" },
                              { value: "srs", label: "srs" },
                            ]}
                            onChange={(v) => update((f) => {
                              f.laser.hotE.channels[i] = makeHotEChannel(
                                v as "cone" | "tpd" | "srs",
                              );
                            })}
                          />
                          <NumInput
                            label={m.form.hoteNcFraction}
                            value={ch.captureNcFraction}
                            onChange={(n) => update((f) => {
                              f.laser.hotE.channels[i].captureNcFraction = n ?? 0.25;
                            })}
                            hint={m.form.hoteCaptureNcHint}
                          />
                          <NumInput
                            label={m.form.hoteTHot}
                            value={ch.THotEV}
                            onChange={(n) => update((f) => {
                              f.laser.hotE.channels[i].THotEV = n ?? 5.0e4;
                            })}
                          />
                          {ch.mechanism === "tpd" ? (
                            <>
                              <NumInput
                                label="tpd_theta_deg [deg]"
                                value={ch.tpdThetaDeg}
                                onChange={(n) => update((f) => {
                                  f.laser.hotE.channels[i].tpdThetaDeg = n ?? 45;
                                })}
                                hint={m.form.hoteTpdThetaHint}
                              />
                              <NumInput
                                label="tpd_delta_deg [deg]"
                                value={ch.tpdDeltaDeg}
                                onChange={(n) => update((f) => {
                                  f.laser.hotE.channels[i].tpdDeltaDeg = n ?? 10;
                                })}
                                hint={m.form.hoteTpdDeltaHint}
                              />
                            </>
                          ) : (
                            <NumInput
                              label={m.form.hoteTheta}
                              value={ch.thetaDivDeg}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].thetaDivDeg = n ?? 60;
                              })}
                            />
                          )}
                          {form.laser.hotE.etaEvolution === "legacy" ? (
                            <>
                              <SelectField
                                label={m.form.hoteEtaMode}
                                value={ch.etaMode}
                                options={[
                                  { value: "constant", label: m.form.hoteEtaConstant },
                                  { value: "table", label: m.form.hoteEtaTable },
                                ]}
                                onChange={(v) => update((f) => {
                                  f.laser.hotE.channels[i].etaMode = v as "constant" | "table";
                                })}
                              />
                              {ch.etaMode === "constant" ? (
                                <NumInput
                                  label={m.form.hoteEta}
                                  value={ch.etaHot}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].etaHot = n ?? 0;
                                  })}
                                />
                              ) : (
                                <WaveformEditor
                                  points={ch.etaPoints}
                                  onChange={(pts) => update((f) => {
                                    f.laser.hotE.channels[i].etaPoints = pts;
                                  })}
                                  yLabel={m.form.hoteEtaAxis}
                                />
                              )}
                            </>
                          ) : (
                            <>
                              <NumInput
                                label="eval_nc_fraction"
                                value={ch.evalNcFraction}
                                onChange={(n) => update((f) => {
                                  f.laser.hotE.channels[i].evalNcFraction = n ?? 0.25;
                                })}
                                hint={m.form.hoteEvalNcHint}
                              />
                              <details>
                                <summary>{m.form.hoteModelAdvanced}</summary>
                                <NumInput
                                  label="threshold_multiplier"
                                  value={ch.thresholdMultiplier}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].thresholdMultiplier = n ?? 1;
                                  })}
                                />
                                <NumInput
                                  label="eta_inf"
                                  value={ch.etaInf}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].etaInf = n ?? 0.01;
                                  })}
                                />
                                <NumInput
                                  label="eta_hard_cap"
                                  value={ch.etaHardCap}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].etaHardCap = n ?? 0.03;
                                  })}
                                />
                                <NumInput
                                  label="shape_coefficient"
                                  value={ch.shapeCoefficient}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].shapeCoefficient = n ?? 1;
                                  })}
                                />
                                <SelectField
                                  label="relaxation_model"
                                  value={ch.relaxationModel}
                                  options={[
                                    { value: "vu2012", label: "vu2012" },
                                    { value: "fixed", label: "fixed" },
                                  ]}
                                  onChange={(v) => update((f) => {
                                    f.laser.hotE.channels[i].relaxationModel = v as "vu2012" | "fixed";
                                  })}
                                />
                                <NumInput
                                  label="relaxation_tau_s"
                                  value={ch.relaxationTauS}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].relaxationTauS = n ?? 6.0e-12;
                                  })}
                                  hint={m.form.hoteTauHint}
                                />
                                <NumInput
                                  label="relaxation_tau_min_s"
                                  value={ch.relaxationTauMinS}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].relaxationTauMinS = n ?? 3.0e-12;
                                  })}
                                  hint={m.form.hoteTauHint}
                                />
                                <NumInput
                                  label="relaxation_tau_max_s"
                                  value={ch.relaxationTauMaxS}
                                  onChange={(n) => update((f) => {
                                    f.laser.hotE.channels[i].relaxationTauMaxS = n ?? 1.0e-11;
                                  })}
                                  hint={m.form.hoteTauHint}
                                />
                              </details>
                            </>
                          )}
                          <details>
                            <summary>{m.form.hoteAdvanced}</summary>
                            <NumInput
                              int
                              label={m.form.hoteGroups}
                              value={ch.nEnergyGroups}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].nEnergyGroups = n ?? 30;
                              })}
                            />
                            <NumInput
                              label={m.form.hoteEMin}
                              value={ch.EMinOverTh}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].EMinOverTh = n ?? 0.2;
                              })}
                            />
                            <NumInput
                              label={m.form.hoteEMax}
                              value={ch.EMaxOverTh}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].EMaxOverTh = n ?? 8;
                              })}
                            />
                            <NumInput
                              int
                              label={m.form.hoteNMu}
                              value={ch.nMu}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].nMu = n ?? 6;
                              })}
                            />
                            <NumInput
                              int
                              label={m.form.hoteNPhi}
                              value={ch.nPhi}
                              onChange={(n) => update((f) => {
                                f.laser.hotE.channels[i].nPhi = n ?? 8;
                              })}
                            />
                          </details>
                        </div>
                      ))}
                      {form.laser.hotE.channels.length < 4 && (
                        <Button
                          onClick={() => update((f) => {
                            f.laser.hotE.channels.push(
                              makeHotEChannel(
                                f.laser.hotE.etaEvolution === "model" ? "srs" : "cone",
                              ),
                            );
                          })}
                        >
                          {m.form.hoteChannelAdd}
                        </Button>
                      )}
                    </>
                  )}
                </>
              )}
            </>
          )}
          {driveMode(form) === "radiation_tr" && (
            <>
              <h2 className="mt-3 text-sm font-semibold">{m.form.driveTrTitle}</h2>
              <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.driveTrNote}</p>
              <SelectField
                label={m.form.marshakMode}
                value={form.radiation.marshakMode}
                options={[
                  { value: "constant", label: m.form.marshakConstant },
                  { value: "table", label: m.form.waveformTable },
                ]}
                onChange={(v) => update((f) => { f.radiation.marshakMode = v as never; })}
              />
              {form.radiation.marshakMode === "constant" ? (
                <NumInput
                  label={m.form.marshakTr}
                  value={form.radiation.marshakTrEV}
                  onChange={(n) => update((f) => { f.radiation.marshakTrEV = n ?? 0; })}
                />
              ) : (
                <WaveformEditor
                  points={form.radiation.marshakPoints}
                  onChange={(pts) => update((f) => { f.radiation.marshakPoints = pts; })}
                  yLabel={m.form.trPointV}
                  importHint={m.form.wfImportHintTr}
                />
              )}
            </>
          )}
          {driveMode(form) === "pressure" && (
            <>
              <SelectField
                label={m.form.pressureDriveMode}
                value={form.hydro.boundaryPressure.mode}
                options={[
                  { value: "constant", label: m.form.pressureDriveConstant },
                  { value: "table", label: m.form.pressureDriveTable },
                ]}
                onChange={(v) => update((f) => { f.hydro.boundaryPressure.mode = v as never; })}
              />
              {form.hydro.boundaryPressure.mode === "constant" ? (
                <QInput
                  label={m.form.pressureDriveValue}
                  kind="pressure"
                  value={form.hydro.boundaryPressure.value}
                  onChange={(q) => update((f) => { f.hydro.boundaryPressure.value = q; })}
                />
              ) : (
                <WaveformEditor
                  points={form.hydro.boundaryPressure.points}
                  onChange={(p) => update((f) => { f.hydro.boundaryPressure.points = p; })}
                  yLabel={m.form.pressureDriveVMbar}
                />
              )}
              <p className="text-xs" style={{ color: "var(--fg-secondary)", maxWidth: "26rem" }}>
                {m.form.pressureDriveHint}
              </p>
            </>
          )}
        </>
    </div>
  );
}

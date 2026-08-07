import { t } from "../../i18n";
import { currentProfile, useApp } from "../../store";
import { BoundsInput, MultiCheckField, NumInput, QInput, SelectField, SwitchField } from "../fields";
import { Button, FieldGroup } from "@tenryu-common/ui/kit";
import WaveformEditor from "../WaveformEditor";
import { conductionFLimDefault } from "../../core/deck/formState";

export default function PhysicsSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const fetchTmatGroupBounds = useApp((s) => s.fetchTmatGroupBounds);
  const tmatBoundsBusy = useApp((s) => s.tmatBoundsBusy);
  const tmatBoundsError = useApp((s) => s.tmatBoundsError);
  const profile = useApp((s) => currentProfile(s));
  const tmatSource = form.materials.find(
    (mm) => mm.opacityModel === "tmat" && mm.opacityFile.trim() !== "",
  );
  return (
    <div className="form-surface">
      <h1 className="mb-2 text-base font-semibold">{m.form.radiationTitle}</h1>
      <div className="field-group-grid">
      <FieldGroup title={m.form.radiationTitle}>
      <SwitchField
        label={m.form.radEnabled}
        checked={form.radiation.enabled}
        onChange={(b) => update((f) => { f.radiation.enabled = b; })}
      />
      {form.radiation.enabled && (
        <>
          <SelectField
            label={m.form.radMode}
            value={form.radiation.mode}
            options={[
              { value: "multigroup_diffusion", label: m.form.radModeFld },
              { value: "sn_transport", label: m.form.radModeSn },
            ]}
            onChange={(v) => update((f) => { f.radiation.mode = v as never; })}
          />
          <NumInput
            int
            label={m.form.groups}
            value={form.radiation.groups}
            onChange={(n) => update((f) => { f.radiation.groups = n ?? 1; })}
          />
          <BoundsInput
            label={m.form.groupBounds}
            value={form.radiation.groupBoundsEV}
            onChange={(ns) => update((f) => { f.radiation.groupBoundsEV = ns; })}
          />
          <div className="flex items-center gap-2">
            <Button
              disabled={tmatBoundsBusy || profile === null || tmatSource === undefined}
              title={
                profile === null
                  ? m.remoteFs.needProfile
                  : tmatSource === undefined
                    ? m.tmatBounds.noTmatMaterial
                    : tmatSource.opacityFile
              }
              onClick={() => void fetchTmatGroupBounds()}
            >
              {tmatBoundsBusy ? m.server.testing : m.tmatBounds.fetch}
            </Button>
            {tmatSource !== undefined && (
              <span className="text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
                {tmatSource.opacityFile.split("/").pop()}
              </span>
            )}
          </div>
          {tmatBoundsError !== null && (
            <p className="text-xs" style={{ color: "var(--err)" }}>
              {tmatBoundsError === "NO_PROFILE" ? m.validate.noProfile : tmatBoundsError}
            </p>
          )}
          <SelectField
            label={m.form.outerBc}
            value={form.radiation.outerR}
            options={[
              { value: "vacuum", label: m.form.bcVacuum },
              { value: "reflect", label: m.form.bcReflect },
              { value: "marshak", label: m.form.bcMarshak },
            ]}
            onChange={(v) => update((f) => { f.radiation.outerR = v as never; })}
          />
          {form.radiation.outerR === "marshak" && (
            <>
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
          {form.radiation.mode === "sn_transport" && (
            <SelectField
              label={m.form.snNAngles}
              value={String(form.radiation.snNAngles)}
              options={["2", "4", "8", "16", "32"].map((value) => ({ value, label: value }))}
              onChange={(v) => update((f) => { f.radiation.snNAngles = Number(v); })}
            />
          )}
          {form.main.dimension === "2D_RZ" && (
            <SelectField
              label={m.form.zBc}
              value={form.radiation.zBc}
              options={[
                { value: "vacuum", label: m.form.bcVacuum },
                { value: "reflect", label: m.form.bcReflect },
              ]}
              onChange={(v) => update((f) => { f.radiation.zBc = v as never; })}
            />
          )}
        </>
      )}
      </FieldGroup>
      <FieldGroup title={m.form.conductionTitle}>
      <SwitchField
        label={m.form.condEnabled}
        checked={form.conduction.enabled}
        onChange={(b) => update((f) => { f.conduction.enabled = b; })}
      />
      {form.conduction.enabled && (
        <>
          {form.main.dimension === "1D_SPH" && form.main.temperatureModel === "2T" && (
            <SelectField
              label={m.form.snbModel}
              value={form.conduction.nonlocalModel}
              options={[
                { value: "none", label: m.form.snbModelNone },
                { value: "snb", label: m.form.snbModelSnb },
              ]}
              onChange={(v) => update((f) => { f.conduction.nonlocalModel = v as never; f.conduction.fLim = conductionFLimDefault(v as "none" | "snb"); })}
            />
          )}
          <NumInput
            label={m.form.condFlim}
            value={form.conduction.fLim}
            onChange={(n) => update((f) => { f.conduction.fLim = n ?? conductionFLimDefault(f.conduction.nonlocalModel); })}
          />
          <SwitchField
            label={m.form.condIon}
            checked={form.conduction.ionConduction}
            onChange={(b) => update((f) => { f.conduction.ionConduction = b; })}
          />
          {form.main.dimension === "1D_SPH" && form.main.temperatureModel === "2T" &&
            form.conduction.nonlocalModel === "snb" && (
                <details>
                  <summary>{m.form.snbAdvanced}</summary>
                  <NumInput
                    int
                    label={m.form.snbGroups}
                    value={form.conduction.snbNGroups}
                    onChange={(n) => update((f) => { f.conduction.snbNGroups = n ?? 24; })}
                  />
                  <NumInput
                    label={m.form.snbEMax}
                    value={form.conduction.snbEMaxOverTe}
                    onChange={(n) => update((f) => { f.conduction.snbEMaxOverTe = n ?? 20; })}
                  />
                  <SelectField
                    label={m.form.snbMfp}
                    value={form.conduction.snbMfp}
                    options={[
                      { value: "geometric_r2", label: m.form.snbMfpGeom },
                      { value: "original", label: m.form.snbMfpOriginal },
                    ]}
                    onChange={(v) => update((f) => { f.conduction.snbMfp = v as never; })}
                  />
                  <SelectField
                    label={m.form.snbEfield}
                    value={form.conduction.snbEfield}
                    options={[
                      { value: "none", label: m.form.snbEfieldNone },
                      { value: "local", label: m.form.snbEfieldLocal },
                    ]}
                    onChange={(v) => update((f) => { f.conduction.snbEfield = v as never; })}
                  />
                  <NumInput
                    int
                    label={m.form.snbPicardIters}
                    value={form.conduction.snbPicardMaxIters}
                    onChange={(n) => update((f) => { f.conduction.snbPicardMaxIters = n ?? 8; })}
                  />
                  <NumInput
                    label={m.form.snbPicardRtol}
                    value={form.conduction.snbPicardRtol}
                    onChange={(n) => update((f) => { f.conduction.snbPicardRtol = n ?? 0.01; })}
                  />
                </details>
              )}
        </>
      )}
      </FieldGroup>
      <FieldGroup title={m.form.hydroTitle}>
      <SwitchField
        label={m.form.hydroEnabled}
        checked={form.hydro.enabled}
        onChange={(b) => update((f) => { f.hydro.enabled = b; })}
      />
      {form.main.dimension === "1D_SPH" && (
        <SelectField
          label={m.form.hydroBc}
          value={form.hydro.boundary1d}
          options={[
            { value: "free", label: m.form.hydroBcFree },
            { value: "reflect", label: m.form.hydroBcReflect },
            { value: "pressure", label: m.form.hydroBcPressure },
          ]}
          onChange={(v) => update((f) => { f.hydro.boundary1d = v as never; })}
        />
      )}
      <NumInput
        label={m.form.hydroTStart}
        value={form.hydro.tStartEV}
        onChange={(n) => update((f) => { f.hydro.tStartEV = n ?? 0; })}
      />
      </FieldGroup>
      {form.hydro.enabled && form.main.dimension === "1D_SPH" && (
        <FieldGroup title={m.form.pviscTitle}>
          <SwitchField
            label={m.form.pviscEnable}
            checked={form.hydro.plasmaVisc.enabled}
            onChange={(b) => update((f) => { f.hydro.plasmaVisc.enabled = b; })}
          />
          {form.hydro.plasmaVisc.enabled && (
            <>
              <SelectField
                label={m.form.pviscModel}
                value={form.hydro.plasmaVisc.model}
                options={[
                  { value: "braginskii", label: m.form.pviscModelBrag },
                  { value: "constant", label: m.form.pviscModelConst },
                ]}
                onChange={(v) => update((f) => { f.hydro.plasmaVisc.model = v as never; })}
              />
              <SelectField
                label={m.form.pviscSpecies}
                value={form.hydro.plasmaVisc.species}
                options={[
                  { value: "ion", label: m.form.pviscSpeciesIon },
                  { value: "electron", label: m.form.pviscSpeciesElectron },
                  { value: "both", label: m.form.pviscSpeciesBoth },
                ]}
                onChange={(v) => update((f) => { f.hydro.plasmaVisc.species = v as never; })}
              />
              {form.hydro.plasmaVisc.model === "constant" && (
                <NumInput
                  label={m.form.pviscEtaConst}
                  value={form.hydro.plasmaVisc.etaConst}
                  onChange={(n) => update((f) => { f.hydro.plasmaVisc.etaConst = n ?? 0; })}
                />
              )}
              <details>
                <summary>{m.form.pviscAdvanced}</summary>
                <NumInput
                  label={m.form.pviscEta0Scale}
                  value={form.hydro.plasmaVisc.eta0Scale}
                  onChange={(n) => update((f) => { f.hydro.plasmaVisc.eta0Scale = n ?? 1; })}
                />
                <NumInput
                  label={m.form.pviscMfpCap}
                  value={form.hydro.plasmaVisc.mfpCapCells}
                  onChange={(n) => update((f) => { f.hydro.plasmaVisc.mfpCapCells = n ?? 20; })}
                />
                <NumInput
                  label={m.form.pviscLnLambda}
                  value={form.hydro.plasmaVisc.lnLambdaFixed}
                  onChange={(n) => update((f) => { f.hydro.plasmaVisc.lnLambdaFixed = n ?? 0; })}
                />
                <NumInput
                  label={m.form.pviscDtSafety}
                  value={form.hydro.plasmaVisc.dtSafety}
                  onChange={(n) => update((f) => { f.hydro.plasmaVisc.dtSafety = n ?? 0.3; })}
                />
              </details>
            </>
          )}
        </FieldGroup>
      )}
      {form.main.dimension === "1D_SPH" && form.main.geometry1d === "spherical" && (
        <FieldGroup title={m.form.burnTitle}>
          <SwitchField
            label={m.form.burnEnable}
            checked={form.burn.enabled}
            onChange={(b) => update((f) => { f.burn.enabled = b; })}
          />
          {form.burn.enabled && (
            <>
              <SwitchField
                label={m.form.burnFuelDT}
                checked={form.burn.fuels.DT}
                onChange={(b) => update((f) => { f.burn.fuels.DT = b; })}
              />
              <SwitchField
                label={m.form.burnFuelDD}
                checked={form.burn.fuels.DD}
                onChange={(b) => update((f) => { f.burn.fuels.DD = b; })}
              />
              <SwitchField
                label={m.form.burnFuelD3He}
                checked={form.burn.fuels.D3He}
                onChange={(b) => update((f) => { f.burn.fuels.D3He = b; })}
              />
              <SelectField
                label={m.form.burnScheme}
                value={form.burn.scheme}
                options={[
                  { value: "fraley", label: m.form.burnSchemeFraley },
                  { value: "diffusion", label: m.form.burnSchemeDiffusion },
                  { value: "mc", label: m.form.burnSchemeMc },
                ]}
                onChange={(v) => update((f) => { f.burn.scheme = v as never; })}
              />
              <SelectField
                label={m.form.burnScreening}
                value={form.burn.screening}
                options={[
                  { value: "none", label: m.form.burnScreeningNone },
                  { value: "salpeter", label: m.form.burnScreeningSalpeter },
                  { value: "chugunov_dewitt", label: m.form.burnScreeningCd },
                ]}
                onChange={(v) => update((f) => { f.burn.screening = v as never; })}
              />
              <MultiCheckField
                label={m.form.burnFuelMaterials}
                options={[...new Set(form.materials.map((mat) => mat.name).filter((n) => n.length > 0))]}
                selected={form.burn.fuelMaterials.split(",").map((s) => s.trim()).filter((s) => s.length > 0)}
                onChange={(next) => update((f) => { f.burn.fuelMaterials = next.join(","); })}
                hint={m.form.burnFuelMaterialsHint}
                undefinedSuffix={m.form.burnFuelMaterialsUndefined}
              />
              <NumInput
                label={m.form.burnXD}
                value={form.burn.xD}
                onChange={(n) => update((f) => { f.burn.xD = n ?? 0; })}
                hint={m.form.burnXFractionHint}
              />
              <NumInput
                label={m.form.burnXT}
                value={form.burn.xT}
                onChange={(n) => update((f) => { f.burn.xT = n ?? 0; })}
              />
              <NumInput
                label={m.form.burnXHe3}
                value={form.burn.xHe3}
                onChange={(n) => update((f) => { f.burn.xHe3 = n ?? 0; })}
              />
              <SwitchField
                label={m.form.burnNeutronHeating}
                checked={form.burn.neutronHeating}
                onChange={(b) => update((f) => { f.burn.neutronHeating = b; })}
              />
              {form.burn.neutronHeating && (
                <NumInput
                  int
                  label={m.form.burnNMu}
                  value={form.burn.neutronHeatingNMu}
                  onChange={(n) => update((f) => { f.burn.neutronHeatingNMu = n ?? 16; })}
                />
              )}
              <details>
                <summary>{m.form.burnAdvanced}</summary>
                <NumInput
                  label={m.form.burnTFloor}
                  value={form.burn.TFloorKeV}
                  onChange={(n) => update((f) => { f.burn.TFloorKeV = n ?? 0.2; })}
                />
                {form.burn.scheme === "mc" && (
                  <NumInput
                    int
                    label={m.form.burnMcParticles}
                    value={form.burn.mcParticlesPerCell}
                    onChange={(n) => update((f) => { f.burn.mcParticlesPerCell = n ?? 16; })}
                  />
                )}
                {form.burn.scheme === "diffusion" && (
                  <>
                    <NumInput
                      int
                      label={m.form.burnDiffGroups}
                      value={form.burn.diffusionGroups}
                      onChange={(n) => update((f) => { f.burn.diffusionGroups = n ?? 30; })}
                    />
                    <NumInput
                      label={m.form.burnDiffEMin}
                      value={form.burn.diffusionEMinKeV}
                      onChange={(n) => update((f) => { f.burn.diffusionEMinKeV = n ?? 20; })}
                    />
                  </>
                )}
                <SelectField
                  label={m.form.burnPartition}
                  value={form.burn.partition}
                  options={[
                    { value: "li_petrasso", label: m.form.burnPartitionLp },
                    { value: "fraley", label: m.form.burnPartitionFraley },
                  ]}
                  onChange={(v) => update((f) => { f.burn.partition = v as never; })}
                />
                <NumInput
                  label={m.form.burnSourceLimit}
                  value={form.burn.explicitSourceLimit}
                  onChange={(n) => update((f) => { f.burn.explicitSourceLimit = n ?? 0.2; })}
                />
                <NumInput
                  label={m.form.burnEpsDeplete}
                  value={form.burn.epsDeplete}
                  onChange={(n) => update((f) => { f.burn.epsDeplete = n ?? 0.1; })}
                />
                <NumInput
                  int
                  label={m.form.burnSubcycle}
                  value={form.burn.subcycleMax}
                  onChange={(n) => update((f) => { f.burn.subcycleMax = n ?? 64; })}
                />
                <NumInput
                  label={m.form.burnVf}
                  value={form.burn.vfThreshold}
                  onChange={(n) => update((f) => { f.burn.vfThreshold = n ?? 1e-3; })}
                />
              </details>
            </>
          )}
        </FieldGroup>
      )}
      <FieldGroup title={m.form.numericsTitle}>
      <QInput
        label={m.form.dtInitial}
        kind="time"
        value={form.numerics.dtInitial}
        onChange={(q) => update((f) => { f.numerics.dtInitial = q; })}
      />
      <QInput
        label={m.form.dtMax}
        kind="time"
        value={form.numerics.dtMax}
        onChange={(q) => update((f) => { f.numerics.dtMax = q; })}
      />
      <QInput
        label={m.form.dtMin}
        kind="time"
        value={form.numerics.dtMin}
        onChange={(q) => update((f) => { f.numerics.dtMin = q; })}
      />
      <NumInput
        label={m.form.growthFactor}
        value={form.numerics.growthFactor}
        onChange={(n) => update((f) => { f.numerics.growthFactor = n ?? 0; })}
      />
      </FieldGroup>
      <FieldGroup title={m.form.floorsTitle}>
      <NumInput
        label={m.form.rhoFloor}
        value={form.numerics.floors.rhoFloorGcc}
        onChange={(n) => update((f) => { f.numerics.floors.rhoFloorGcc = n ?? 0; })}
      />
      <NumInput
        label={m.form.teFloor}
        value={form.numerics.floors.TeFloorEV}
        onChange={(n) => update((f) => { f.numerics.floors.TeFloorEV = n ?? 0; })}
      />
      <NumInput
        label={m.form.tiFloor}
        value={form.numerics.floors.TiFloorEV}
        onChange={(n) => update((f) => { f.numerics.floors.TiFloorEV = n ?? 0; })}
      />
      </FieldGroup>
      </div>
    </div>
  );
}

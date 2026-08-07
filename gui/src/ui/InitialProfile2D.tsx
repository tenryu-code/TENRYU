import { useMemo, useState } from "react";
import { logLine } from "@tenryu-common/core/applog";
import type { FormState } from "../core/deck/formState";
import { sampleInitialProfile, type ProfileField } from "../core/initialProfile";
import { computeLocalMeshPreview } from "../core/localMeshPreview";
import { t } from "../i18n";
import { NumInput, SelectField } from "./fields";

const WIDTH = 560;
const HEIGHT = 220;
const MARGIN = 12;

export default function InitialProfile2D({ form }: { form: FormState }) {
  const m = t();
  const rNodes = useMemo(() => {
    try {
      return computeLocalMeshPreview(form)?.rNodes ?? null;
    } catch (e) {
      logLine("error", `preview compute failed: ${e}`);
      return null;
    }
  }, [form]);
  const [theta, setTheta] = useState(Math.PI / 2);
  const [field, setField] = useState<ProfileField>("rho");
  const prof = sampleInitialProfile(form, theta, field);

  if (prof === null || !Number.isFinite(prof.sEnd) || prof.sEnd <= 0) {
    return (
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
        {m.profile2d.unavailable}
      </p>
    );
  }

  const thetaDeg = (theta * 180) / Math.PI;
  const maxValue = Math.max(...prof.samples.map((sample) => sample.value));
  const yMax = maxValue > 0 ? 1.05 * maxValue : 1;
  const x = (s: number) => MARGIN + (s * (WIDTH - 2 * MARGIN)) / prof.sEnd;
  const y = (value: number) => MARGIN + ((yMax - value) * (HEIGHT - 2 * MARGIN)) / yMax;
  const unit = field === "rho" ? "g/cm3" : "eV";
  const showNodeTicks =
    form.mesh.logicalMesh2d === "spherical_polar_halfplane" &&
    rNodes !== null;

  return (
    <div className="mt-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-start gap-2">
        <SelectField
          label=""
          value={field}
          options={[
            { value: "rho", label: m.profile2d.fieldRho },
            { value: "Te", label: m.profile2d.fieldTe },
            { value: "Ti", label: m.profile2d.fieldTi },
          ]}
          onChange={(value) => setField(value as ProfileField)}
        />
        <NumInput
          label={m.profile2d.angleDeg}
          value={thetaDeg}
          onChange={(value) => {
            const degrees = Math.max(0, Math.min(180, value !== null && Number.isFinite(value) ? value : 0));
            setTheta((degrees * Math.PI) / 180);
          }}
        />
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width={WIDTH}
        height={HEIGHT}
        style={{ background: "#0b0e14", borderRadius: 8 }}
      >
        {prof.regionEdges.map((edge) => (
          <line
            key={edge}
            x1={x(edge)}
            y1={MARGIN}
            x2={x(edge)}
            y2={HEIGHT - MARGIN}
            stroke="#ff9f43"
            strokeWidth={1.5}
            strokeDasharray="6 4"
          />
        ))}
        {showNodeTicks &&
          rNodes.map((node) => (
            <line
              key={node}
              x1={x(node)}
              y1={HEIGHT - MARGIN - 6}
              x2={x(node)}
              y2={HEIGHT - MARGIN}
              stroke="#3d4a63"
              strokeWidth={1}
            />
          ))}
        <polyline
          points={prof.samples.map((sample) => `${x(sample.s)},${y(sample.value)}`).join(" ")}
          stroke="#4c8dff"
          strokeWidth={1.5}
          fill="none"
        />
        <text x={WIDTH / 2} y={HEIGHT - 2} textAnchor="middle" fill="var(--fg-secondary)" fontSize={10}>
          s [cm]
        </text>
        <text x={MARGIN} y={10} fill="var(--fg-secondary)" fontSize={10}>
          {field} [{unit}]
        </text>
      </svg>
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
        θ = {thetaDeg}° · sEnd = {prof.sEnd.toPrecision(4)} cm
      </p>
    </div>
  );
}

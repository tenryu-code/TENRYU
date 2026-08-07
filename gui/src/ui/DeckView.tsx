import { useApp } from "../store";
import BasicSection from "./sections/BasicSection";
import LaserSection from "./sections/LaserSection";
import MaterialsSection from "./sections/MaterialsSection";
import MeshSection from "./sections/MeshSection";
import OutputSection from "./sections/OutputSection";
import PhysicsSection from "./sections/PhysicsSection";
import PresetsSection from "./sections/PresetsSection";

export default function DeckView() {
  const section = useApp((s) => s.section);
  switch (section) {
    case "basic":
      return <BasicSection />;
    case "mesh":
      return <MeshSection />;
    case "materials":
      return <MaterialsSection />;
    case "physics":
      return <PhysicsSection />;
    case "laser":
      return <LaserSection />;
    case "output":
      return <OutputSection />;
    case "presets":
      return <PresetsSection />;
  }
}

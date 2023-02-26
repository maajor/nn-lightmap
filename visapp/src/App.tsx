import { extend } from "@react-three/fiber";
import { Stats, OrbitControls, Float } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { NnLightmapMaterial } from "./NnLightmapMaterial";
import { Scene } from "./Scene";

extend({ NnLightmapMaterial });

function App() {
  return (
    <div id="canvas-container">
      <Canvas camera={{ position: [0.6, 0.4, 1], fov: 48.5 }}>
        <OrbitControls target={[0, 0.1, 0]} />
        <Stats />
        <group position={[0, 0, 0]}>
          <Float
            position={[0, 0, 0]}
            speed={2}
            rotationIntensity={0.1}
            floatIntensity={0.1}
          >
            <Scene />
          </Float>
        </group>
      </Canvas>
    </div>
  );
}

export default App;

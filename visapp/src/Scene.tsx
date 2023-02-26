import { extend } from "@react-three/fiber";
import { useGLTF, useTexture } from "@react-three/drei";
import { LinearEncoding, Texture } from "three";
import { NnLightmapMaterial } from "./NnLightmapMaterial";

extend({ NnLightmapMaterial });

export const Scene = () => {
  const gltf = useGLTF("/text/bake_sample.glb");
  const [t0, t1] = useTexture(
    ["/text/lightmap_0.0.webp", "/text/lightmap_1.0.webp"],
    (texs) => {
      for (const tex of texs as Texture[]) {
        tex.encoding = LinearEncoding;
        tex.needsUpdate = true;
      }
    }
  );
  return (
    // @ts-ignore
    <mesh geometry={gltf.nodes.Object001.geometry}>
      <nnLightmapMaterial lm0={t0} lm1={t1} />
    </mesh>
  );
};

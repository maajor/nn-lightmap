import { extend } from '@react-three/fiber';
import { useGLTF, useTexture } from '@react-three/drei'
import { LinearEncoding, Texture } from 'three'
import { NnLightmapMaterial } from './NnLightmapMaterial';

extend({ NnLightmapMaterial });

export const Scene = () => {
    const gltf = useGLTF('/bake_sample.glb')
    const [t0, t1, t2, t3] = useTexture(['/lightmap_0.0.png', '/lightmap_1.0.png', '/lightmap_2.0.png', '/lightmap_3.0.png'],
        (texs) => {
            for (const tex of (texs as Texture[])) {
                tex.encoding = LinearEncoding;
                tex.needsUpdate = true;
            }
        });
    return (
        // @ts-ignore
        <mesh geometry={gltf.nodes.Object001.geometry}>
            <nnLightmapMaterial lm0={t0} lm1={t1} lm2={t2} lm3={t3}/>
        </mesh>

    )
}
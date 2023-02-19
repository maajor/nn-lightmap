import { useState } from 'react'
import { extend } from '@react-three/fiber';
import { Stats, OrbitControls, useGLTF, useTexture } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Color } from 'three'
import { NnLightmapMaterial } from './NnLightmapMaterial';

extend({ NnLightmapMaterial });

export const Scene = () => {
    const gltf = useGLTF('/bake_sample.glb')
    const [t0, t1, t2, t3, t4, t5, t6, t7] = useTexture(['/lightmap_0.0.png', '/lightmap_1.0.png', '/lightmap_2.0.png', '/lightmap_3.0.png', '/lightmap_4.0.png', '/lightmap_5.0.png', '/lightmap_6.0.png', '/lightmap_7.0.png']);
    return (
        <mesh geometry={gltf.nodes.Object001.geometry}>
            <nnLightmapMaterial time={1} color={new Color("hotpink")} lm0={t0} lm1={t1} lm2={t2} lm3={t3} lm4={t4} lm5={t5} lm6={t6} lm7={t7} />
        </mesh>

    )
}
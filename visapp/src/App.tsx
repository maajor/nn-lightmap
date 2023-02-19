import { useState } from 'react'
import { extend } from '@react-three/fiber';
import { Stats, OrbitControls, useGLTF, useTexture } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import {Color} from 'three'
import { NnLightmapMaterial } from './NnLightmapMaterial';
import {Scene} from './Scene';

extend({ NnLightmapMaterial });

function App() {

  return (
    <div id="canvas-container">
      <Canvas
        camera={{ position: [10, 10, 10] }}>
        <Scene/>
        <OrbitControls target={[0, 0, 0]} />
        <Stats />
      </Canvas>
    </div>
  )
}

export default App

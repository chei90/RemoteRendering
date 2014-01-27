#ifndef __GLOBALS__
#define __GLOBALS__

#define CASCADES_COUNT 3

#define PI 3.1415926535897
#define PIDIV2 PI/2.0f

SamplerState g_samPoint
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Wrap;
    AddressV = Wrap;
};

cbuffer ViewBuffer : register(b0)
{
    matrix g_view;
    matrix g_invView;
    float4 g_eyePos;
}

cbuffer ProjectionBuffer : register(b1)
{
    matrix g_projection;
    float4 g_viewDistance;
}

cbuffer ModelMatrixBuffer : register(b2)
{
    matrix g_model;
}

cbuffer MaterialBuffer : register(b3)
{
    float4 g_ambientMaterial;
    float4 g_specularMaterial;
    float4 g_diffuseMaterial;
    float g_specularExpoMaterial;
    float g_reflectanceMaterial;
    float g_textureScale;
    float unused;
}

cbuffer CubeMapViewMaticesBuffer : register(b4)
{
    matrix g_cubeViewMatrices[6];
}

cbuffer LightSettingsBuffer : register(b5)
{
    float4 g_lightColorRadius;
    float4 g_lightPos;
    float4 g_lightViewDirAngle;
    int4 g_castShadow;
}

cbuffer FontBuffer : register(b6)
{
    float2 g_fontTextureOffset;
    float2 g_fontTextureScale;
    float2 g_fontPositionOffset;
    float2 g_fontPositionScale;
}

cbuffer BoundingGeoBuffer : register(b7)
{
    float g_sphereScale;
    float3 g_sphereMiddle;
}

cbuffer ActorIdBuffer : register(b8)
{
    uint4 g_actorId;
}

cbuffer SelectedActorBuffer : register(b9)
{
    uint4 g_selectedActorId;
}

cbuffer ColorBuffer : register(b10)
{
    float4 g_color;
}

cbuffer HasNormalMapBuffer : register(b11)
{
    float4 g_hasNormalMap;
}

cbuffer CSMLightingBuffer : register(b12)
{
    matrix g_lightView;
    matrix g_IlightView;
    matrix g_lightProjection[CASCADES_COUNT];
    float4 g_CSMlightPos;
    float4 g_distances;
}

SamplerState g_samplerWrap : register(s0);
SamplerState g_samplerClamp : register(s1);
SamplerState g_samplerAniso : register(s2);
SamplerState g_samplerPoint : register(s3);

Texture2D<float4> g_diffuseColor : register(t0);
Texture2D<float4> g_worldPosDepth : register(t1);
Texture2D<float4> g_normals : register(t2);
Texture2D<float4> g_diffuseMaterialSpecR : register(t3);
Texture2D<float4> g_ambientMaterialSpecG : register(t4);
Texture2D<float4> g_diffuseColorSpecB : register(t5);
TextureCube g_pointLightShadowMap : register(t6);
Texture2D<float4> g_guiTexture : register(t7);
Texture2D<float4> g_normalColor : register(t8);
Texture2D<float4> g_scene : register(t9);

//effect slots 4 and cascaded shadow maps
Texture2D<float4> g_effectSource0 : register(t10);
Texture2D<float4> g_effectSource1 : register(t11);
Texture2D<float4> g_effectSource2 : register(t12);
Texture2D<float4> g_effectSource3 : register(t13);

//Texture2D<float4> g_cascadedShadows[CASCADES_COUNT] : register(t14); //TODO slot

#endif //__GLOBALS__
#include "ShaderGlobals.h"

struct PixelInput 
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

struct PixelOutput 
{
    float4 worldPosDepth : SV_Target0;
    float4 normal : SV_Target1;
    half4 diffMaterialSpecR : SV_Target2;
    half4 ambientMaterialSpecG : SV_Target3;
    half4 diffuseColorSpecB : SV_Target4;
    //float2 specExReflectance : SV_Target5;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
};


PixelInput Sphere_VS(VertexInput input) 
{
    PixelInput op;
    op.position = mul(g_projection, mul(g_view, mul(g_model, float4(g_sphereMiddle + g_sphereScale * input.position, 1))));
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput Sphere_PS(PixelInput input)
{
    PixelOutput op;
    float4 tex = float4(1,1,1,1);

    op.worldPosDepth.xyz = input.position.xyz;
    op.worldPosDepth.w = input.position.w;

    op.normal = float4(0,1,0,0);
    op.diffMaterialSpecR = half4(1,1,1,0);
    op.ambientMaterialSpecG = half4(1,1,1,0);
    op.diffuseColorSpecB = half4(1,1,1,0);

    return op;
}
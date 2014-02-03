#include "ShaderGlobals.h"

struct PixelInput 
{
    float4 position : SV_POSITION;
    float4 worldpos : POSITION0;
    float2 texCoord : TEXCOORD0;
};

struct PixelOutput 
{
    float4 worldPosDepth : SV_Target0;
    float4 normal : SV_Target1;
    half4 diffMaterialSpecR : SV_Target2;
    half4 ambientMaterialSpecG : SV_Target3;
    half4 diffuseColorSpecB : SV_Target4;
    half reflectance : SV_Target5;
    //float2 specExReflectance : SV_Target5;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
    //float3 tangent : TANGENT;
};

PixelInput Sky_VS(VertexInput input) 
{
    PixelInput op;
    op.worldpos = mul(g_model, float4(input.position, 0));
    float4 tmp = mul(g_view, op.worldpos); tmp.w = 1;
    op.position = mul(g_projection, tmp);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput Sky_PS(PixelInput input)
{
    PixelOutput op;
    float2 tc = float2(input.texCoord.x, input.texCoord.y);
    float4 tex = g_diffuseColor.Sample(g_samplerWrap, tc);

    op.worldPosDepth.xyz = input.worldpos.xyz;
    op.worldPosDepth.w = 0;

    op.normal = float4(0,0,0,0);
    op.diffMaterialSpecR = half4(0,0,0,0);
    op.ambientMaterialSpecG = half4(0,0,0,0);
    op.diffuseColorSpecB = 1.25 * half4(tex.xyz, 1);
    op.reflectance = 0;
    return op;
}
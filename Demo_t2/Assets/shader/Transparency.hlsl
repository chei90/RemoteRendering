#include "ShaderGlobals.h"

struct PixelInput 
{
    float2 texCoords : TEXCOORD0;
    float4 position : SV_POSITION;
    float4 world : POSITION1;
    float4 worldView : POSITION2;
    float3 normal : NORMAL0;
   // float3 tangent : TANGENT;
};

struct PixelOutput 
{
    float4 color : SV_Target0;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    //float3 tangent : TANGENT;
};

struct VertexInputInstanced
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    float3 instanceTranslation : TANGENT0;
};

PixelInput TransparencyInstanced_VS(VertexInputInstanced input) 
{
    PixelInput op;
    op.world = mul(g_model, float4(input.position + input.instanceTranslation, 1));
    op.world /= op.world.w;
    op.worldView = mul(g_view, op.world);
    op.position = mul(g_projection, op.worldView);
    op.texCoords = input.texCoord;
    op.normal = mul(g_model, float4(input.normal, 0)).xyz;
    return op;
}


PixelInput Transparency_VS(VertexInput input) 
{
    PixelInput op;
    op.world = mul(g_model, float4(input.position, 1));
    op.world /= op.world.w;
    op.worldView = mul(g_view, op.world);
    op.position = mul(g_projection, op.worldView);
    op.texCoords = input.texCoord;
    op.normal = mul(g_model, float4(input.normal, 0)).xyz;
    return op;
}

PixelOutput Transparency_PS(PixelInput input)
{
    PixelOutput op;
    //TODO
    op.color = half4(1,1,0,1);

    return op;
}
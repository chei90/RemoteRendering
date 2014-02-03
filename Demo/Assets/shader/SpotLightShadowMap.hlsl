#include "ShaderGlobals.h"

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
};

struct VertexOutput 
{
    float4 position : SV_POSITION;
    float3 pos2light : POSITION0;
};

struct PixelOutput 
{
    float4 color : SV_Target;
};

VertexOutput SpotLightShadow_VS(VertexInput input) 
{
    VertexOutput op;
    float4 p = mul(g_model, float4(input.position, 1));
    op.position = mul(g_projection, mul(g_view, p));
    op.pos2light = g_lightPos.xyz - p.xyz;
    return op;
}

PixelOutput SpotLightShadow_PS(VertexOutput input)
{
    PixelOutput op;
    op.color = float4(dot(input.pos2light, input.pos2light), 0, 0, 1);
    return op;
}

//----instanced geometry-----//

struct VertexInput_Instanced
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    float3 instancedTranslation : TANGENT0;
};

VertexOutput SpotLightShadowInstanced_VS(VertexInput_Instanced input) 
{
    VertexOutput op;
    float4 p = mul(g_model, float4(input.position + input.instancedTranslation, 1));
    op.position = mul(g_model, p);
    op.pos2light = g_lightPos.xyz - p.xyz;
    return op;
}
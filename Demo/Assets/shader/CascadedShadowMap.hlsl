#include "ShaderGlobals.h"

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
};

struct VertexInputInstanced 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    float3 translation : TANGENT0;
};

struct VertexOutput 
{
    float4 lightViewPos : NORMAL0;
    float4 position : SV_POSITION;
};

struct PixelOutput 
{
    float4 color : SV_Target;
};

VertexOutput CSM_VS(VertexInput input) 
{
    VertexOutput op;
    op.lightViewPos = mul(g_view, mul(g_model, float4(input.position, 1)));
    op.position = mul(g_projection, op.lightViewPos);
    return op;
}

VertexOutput CSM_Instanced_VS(VertexInputInstanced input) 
{
    VertexOutput op;
    op.lightViewPos = mul(g_view, mul(g_model, float4(input.position + input.translation, 1)));
    op.position = mul(g_projection, op.lightViewPos);
    return op;
}

PixelOutput CSM_PS(VertexOutput input)
{
    PixelOutput op;
    float dist = (120 + input.lightViewPos.z) / g_viewDistance.x;
    op.color = float4(dist, dist * dist, 0, 0);
    float xd = ddx(dist);
    float yd = ddy(dist);
    op.color.y = op.color.y + 0.25 * (xd * xd + yd * yd);
    return op;
}
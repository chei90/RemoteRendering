#include "ShaderGlobals.h"

struct PixelInput 
{
    float4 position : SV_POSITION;
};

struct PixelOutput 
{
    uint actorId : SV_Target0;
    //float4 color : SV_TARGET0;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
};

PixelInput Picking_VS(VertexInput input) 
{
    PixelInput op;
    op.position = mul(g_projection, mul(g_view, mul(g_model, float4(input.position, 1))));
    return op;
}

PixelOutput Picking_PS(PixelInput input)
{
    PixelOutput op;
    op.actorId = g_actorId.x;
    return op;
}
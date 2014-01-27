#include "ShaderGlobals.h"

struct PixelOutput 
{
    float4 worldPosDepth : SV_Target0;
    float4 normal : SV_Target1;
    half4 diffMaterialSpecR : SV_Target2;
    half4 ambientMaterialSpecG : SV_Target3;
    half4 diffuseColorSpecB : SV_Target4;
    //float2 specExReflectance : SV_Target5;
};
/*
struct PixelOutput 
{
    float4 color : SV_Target0;
};
*/

struct VertexInput 
{
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
    float4 translation : TANGENT0;
};

struct PixelInput 
{
    float2 texCoord : TEXCOORD0;
    float4 position : SV_POSITION;
    float4 world : POSITION1;
    float4 worldView : POSITION2;
    float3 normal : NORMAL0;
    float alive : POSITION3;
};

PixelInput Particle_VS(VertexInput input) 
{
    PixelInput op;
    op.world = mul(g_model, float4(input.position, 1)) + float4(input.translation.xyz, 0);
    op.world /= op.world.w;
    op.worldView = mul(g_view, op.world);
    op.position = mul(g_projection, op.worldView);
    op.texCoord = input.texCoord;
    op.normal = input.normal;//mul(g_model, float4(input.normal, 0)).xyz;
    op.alive = input.translation.w;
    return op;
}

PixelOutput Particle_PS(PixelInput input)
{
    float2 tc =  4 * input.texCoord - 2;
    tc.y = - tc.y;
    float f = 0;
    
    if(tc.x < 0)
    {
        f = 2 * tc.x * tc.x - 2 * tc.x * tc.y + tc.y * tc.y - 1;
    }
    else
    {
       f = 2 * tc.x * tc.x + 2 * tc.x * tc.y + tc.y * tc.y - 1;
    }

    if(input.alive < 0.5 || tc.x*tc.x + tc.y*tc.y > 1)//|| f > 0)
    {
        discard;
    }
    
    PixelOutput op;
    //op.color = float4(1,1,0,1);
    
    op.worldPosDepth = input.world;
    op.worldPosDepth.w = length(input.worldView.xyz);

    op.normal = float4(0,0,0,0);
    half scale = 1;
    half4 color = scale * half4(0.9,1,0.1,1); 
    op.diffMaterialSpecR = half4(1,1,1,1);
    op.ambientMaterialSpecG = half4(1,1,1,1);
    op.diffuseColorSpecB = half4(color);
    
    return op;
}
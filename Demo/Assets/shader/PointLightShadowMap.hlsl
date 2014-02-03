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

struct GeoOutput 
{
    float4 position : SV_POSITION;
    float3 pos2light : POSITION0;
    uint RTindex : SV_RenderTargetArrayIndex;
};

struct PixelOutput 
{
    float4 color : SV_Target;
};

VertexOutput RenderCubeMap_VS(VertexInput input) 
{
    VertexOutput op;
    op.position = mul(g_model, float4(input.position, 1));
    op.pos2light = g_lightPos.xyz - op.position.xyz;
    return op;
}

[maxvertexcount(18)]
void RenderCubeMap_GS(triangle VertexOutput input[3], inout TriangleStream<GeoOutput> outstream)
{
    GeoOutput op;
    float4x4 trans = {1,0,0,-g_lightPos.x, 0,1,0,-g_lightPos.y, 0,0,1,-g_lightPos.z, 0,0,0,1};

    [unroll]
    for(int j = 0; j < 6; ++j)
    {
        op.RTindex = j;
        [unroll]
        for(int i = 0; i < 3; ++i)
        {
            op.position = mul(g_projection, mul(mul(g_cubeViewMatrices[j], trans), input[i].position));
            op.pos2light = input[i].pos2light;
            outstream.Append(op);
        }
        outstream.RestartStrip();
    }
}

PixelOutput RenderCubeMap_PS(GeoOutput input)
{
    PixelOutput op;
    op.color = float4(dot(input.pos2light, input.pos2light), 0, 0, 1);
    return op;
}

//-----Particles-----//

struct VertexInput_Particles
{
    float3 position : POSITION0;
    float3 translation : TANGENT;
};

VertexOutput RenderCubeMapParticles_VS(VertexInput_Particles input) 
{
    VertexOutput op;
    op.position = mul(g_model, float4(input.position, 1)) + float4(input.translation, 0);
    op.pos2light = g_lightPos.xyz - op.position.xyz;
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

VertexOutput RenderCubeMapInstanced_VS(VertexInput_Instanced input) 
{
    VertexOutput op;
    op.position = mul(g_model, float4(input.position + input.instancedTranslation, 1));
    op.pos2light = g_lightPos.xyz - op.position.xyz;
    return op;
}
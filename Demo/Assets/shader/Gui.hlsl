#include "ShaderGlobals.h"

struct VertexOutput 
{
    float2 texCoord : TEXCOORD0;
    float4 position : SV_POSITION;
};

struct PixelOutput 
{
    float4 color : SV_Target0;
};

struct VertexInput 
{
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
};

VertexOutput Font_VS(VertexInput input) 
{
    VertexOutput op;
    op.position = float4(input.position, 1);//float4(-1,-1, 0, 0) + float4(g_fontPositionOffset + g_fontPositionScale * input.position.xy, 0, 1);
    op.texCoord = input.texCoord;//g_fontTextureOffset + g_fontTextureScale * input.texCoord;
    return op;
}

PixelOutput Font_PS(VertexOutput input)
{
    PixelOutput op;
    float r = g_guiTexture.Sample(g_samplerClamp, input.texCoord).r;
   // float r = g_fontMap.Load(int3(input.texCoord.x * 128, input.texCoord.y * 140, 0)).x;
    float3 fontColor = g_color.xyz;//float3(1,0,0);
    op.color = float4(fontColor, r);
    
    /*if(r > 0)
    {
        op.color = float4(fontColor,1);
    }
    else
    {
        //op.color = float4(1,0,0,1);
        discard;
    }*/
    return op;
}

VertexOutput GuiDefault_VS(VertexInput input) 
{
    VertexOutput op;
    op.position = float4(input.position.xy, 0, 1);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput GuiDefault_PS(VertexOutput input)
{
    PixelOutput op;
    op.color = g_color;
    return op;
}

PixelOutput GuiTexture_PS(VertexOutput input)
{
    PixelOutput op;
    float4 color = g_guiTexture.Sample(g_samplerClamp, input.texCoord);
    if(color.w < 0.5)
    {
        discard;
    }
    op.color = color * g_color;
    return op;
}
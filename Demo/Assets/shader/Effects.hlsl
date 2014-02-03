#include "ShaderGlobals.h"

struct PixelInput 
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

PixelInput Effect_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput Luminance(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    float lumi = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
    lumi = log(0.01 + lumi);
    op.color = float4(lumi, lumi, lumi, 0);
    op.color.a = 1;
    return op;
}

PixelOutput SampleDiffuseTexture(PixelInput input)
{
    PixelOutput op;
    float4 color = g_diffuseColor.Sample(g_samplerClamp, input.texCoord);
    op.color = color;
    return op;
}

PixelOutput Sample(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    op.color = color;
    return op;
}

static const float weights[7] = {
0.00038771,  
0.01330373,
0.11098164,
0.22508352,
0.11098164,
0.01330373,
0.00038771
};

PixelOutput BlurH(PixelInput input)
{
    float4 color = float4(0, 0, 0, 0);

    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / w, 0);

    float2 fs_in_tex = input.texCoord;
        
    float4 c0 = 0.0000000076834112 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (15.0 + 0.030303)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (15.0 + 0.030303)*texelSize));

    float4 c1 = 0.0000012703239918 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (13.0 + 0.090909)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (13.0 + 0.090909)*texelSize));

    float4 c2 = 0.0000552590936422 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (11.0 + 0.151515)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (11.0 + 0.151515)*texelSize));

    float4 c3 = 0.0009946636855602 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (9.0 + 0.212121)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (9.0 + 0.212121)*texelSize));

    float4 c4 = 0.0089796027168632 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (7.0 + 0.272727)*texelSize)
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (7.0 + 0.272727)*texelSize));

    float4 c5 = 0.0450612790882587 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (5.0 + 0.333333)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (5.0 + 0.333333)*texelSize));

    float4 c6 = 0.1334507111459971 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (3.0 + 0.393939)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (3.0 + 0.393939)*texelSize));

    float4 c7 = 0.2414822392165661 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (1.0 + 0.454545)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (1.0 + 0.454545)*texelSize));

    float4 c8 = 0.1399499340914190 * g_effectSource0.Sample(g_samplerClamp, fs_in_tex);
      
    PixelOutput op;
      op.color = float4(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8);

    return op;
}

PixelOutput BlurV(PixelInput input)
{
    float4 color = float4(0, 0, 0, 0);

    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(0, 1.0 / h);

    float2 fs_in_tex = input.texCoord;
        
    float4 c0 = 0.0000000076834112 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (15.0 + 0.030303)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (15.0 + 0.030303)*texelSize));

    float4 c1 = 0.0000012703239918 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (13.0 + 0.090909)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (13.0 + 0.090909)*texelSize));

    float4 c2 = 0.0000552590936422 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (11.0 + 0.151515)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (11.0 + 0.151515)*texelSize));

    float4 c3 = 0.0009946636855602 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (9.0 + 0.212121)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (9.0 + 0.212121)*texelSize));

    float4 c4 = 0.0089796027168632 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (7.0 + 0.272727)*texelSize)
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (7.0 + 0.272727)*texelSize));

    float4 c5 = 0.0450612790882587 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (5.0 + 0.333333)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (5.0 + 0.333333)*texelSize));

    float4 c6 = 0.1334507111459971 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (3.0 + 0.393939)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (3.0 + 0.393939)*texelSize));

    float4 c7 = 0.2414822392165661 * (g_effectSource0.Sample(g_samplerClamp, fs_in_tex - (1.0 + 0.454545)*texelSize) 
        + g_effectSource0.Sample(g_samplerClamp, fs_in_tex + (1.0 + 0.454545)*texelSize));
    
    float4 c8 = 0.1399499340914190 * g_effectSource0.Sample(g_samplerClamp, fs_in_tex);
      
    PixelOutput op;
      op.color = float4(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8);
    return op;
}
#define FILTER_SIZE 7
PixelOutput VSMBlurV(PixelInput input)
{
    PixelOutput op;
    float4 color = float4(0, 0, 0, 0);
    uint filterSize = FILTER_SIZE;
    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / w, 0);

    float2 fs_in_tex = input.texCoord - texelSize * filterSize / 2;
    
    for(uint i = 0; i < filterSize; ++i)
    {
        color += g_effectSource0.Sample(g_samplerClamp, fs_in_tex + texelSize * i);
    }
    color /= filterSize;
    op.color = color;
    return op;
}

PixelOutput VSMBlurH(PixelInput input)
{
    PixelOutput op;
    float4 color = float4(0, 0, 0, 0);
    uint filterSize = FILTER_SIZE;
    uint w, h, levels;
    g_effectSource0.GetDimensions(0, w, h, levels);
    
    float2 texelSize = float2(1.0 / h, 0);

    float2 fs_in_tex = input.texCoord - texelSize * filterSize / 2;
    
    for(uint i = 0; i < filterSize; ++i)
    {
        color += g_effectSource0.Sample(g_samplerClamp, fs_in_tex + texelSize * i);
    }
    color /= filterSize;
    op.color = color;
    return op;
}

PixelOutput Brightness(PixelInput input)
{
    PixelOutput op;
    op.color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    if(!(op.color.x > 1 || op.color.y > 1 || op.color.z > 1))
    {
        discard;
    }
    return op;
}

PixelOutput ToneMap(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource0.Sample(g_samplerClamp, input.texCoord);
    float4 bright = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    float avgLum = exp(g_effectSource2.Sample(g_samplerClamp, input.texCoord).x);
    
    float lumi = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;

    float key = 0.75; //max(0, 1.5 - 1.5 / (avgLum*0.1 + 1)) + 0.1; //ke = 10 mehr glitzer
    float yr = key * lumi / avgLum;
    float lumiScaled = yr / (1 + yr);

    color += bright;
    //color = color;// * lumiScaled;
    
    //debugging
 
    /*if(input.texCoord.x < 0.5 && input.texCoord.y < 0.5)
    {
        float s = g_effectSource3.Sample(g_samplerClamp, 2 * input.texCoord).x;
        color.y = color.z = color.x = s;
    } */
   
    op.color = color;//pow(abs(color), 1.0 / 2.2);
    return op;
}

#define VLS_SAMPLES 80

PixelOutput LightScattering(PixelInput input)
{
    PixelOutput op;
    float4 color = g_effectSource1.Sample(g_samplerClamp, input.texCoord);
    float4 screenSpaceLightPos = mul(g_projection, mul(g_view, float4(1000 * g_lightPos.xyz, 1))); //scale out
    screenSpaceLightPos /= screenSpaceLightPos.w;
    
    float2 nssl = 0.5f * float2(screenSpaceLightPos.x, screenSpaceLightPos.y) + 0.5f;
    nssl = float2(nssl.x, 1 - nssl.y);
    float2 deltaTexoord = input.texCoord - nssl;
    
    float density = 1;
    
    deltaTexoord *= 1.0f / VLS_SAMPLES * density;
    
    float illumDecay = 1.0f;
    float decay = 0.98f;
    float weight = 1.0f / VLS_SAMPLES;
    float2 tc = input.texCoord;
    for(int i = 0; i < VLS_SAMPLES; ++i)
    {
        tc -= deltaTexoord;
        float depth = g_effectSource0.Sample(g_samplerClamp, tc).w;
        //float4 sample = illumDecay * g_effectSource1.Sample(g_samplerClamp, tc) * weight;
        color = (color * 0.5) * sign(depth);
        illumDecay *= decay;
    }
    
    op.color = color;

    return op;
}


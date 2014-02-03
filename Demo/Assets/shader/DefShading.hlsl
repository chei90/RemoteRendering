#include "ShaderGlobals.h"

#define HIGHLIGHT_PICKED \
    {\
        float scale = 1; \
        op.ambientMaterialSpecG += half4((g_selectedActorId.x == g_actorId.x) * scale, 0, 0, 0); \
        op.diffuseColorSpecB += half4((g_selectedActorId.x == g_actorId.x) * scale, 0, 0, 0); \
    }


struct PixelInput 
{
    float2 texCoords : TEXCOORD0;
    float4 position : SV_POSITION;
    float4 world : POSITION1;
    float4 worldView : POSITION2;
    float3 normal : NORMAL0;
};

struct PixelOutput 
{
    float4 worldPosDepth : SV_Target0;
    float4 normal : SV_Target1;
    half4 diffMaterialSpecR : SV_Target2;
    half4 ambientMaterialSpecG : SV_Target3;
    half4 diffuseColorSpecB : SV_Target4;
    half reflectionStr : SV_Target5;
    //float2 specExReflectance : SV_Target5;
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
    //float3 tangent : TANGENT;
};

PixelInput DefShadingInstanced_VS(VertexInputInstanced input) 
{
    PixelInput op;
    op.world = mul(g_model, float4(input.position,1));
    op.world += float4(input.instanceTranslation, 0);
    op.world /= op.world.w;
    op.worldView = mul(g_view, op.world);
    op.position = mul(g_projection, op.worldView);
    op.texCoords = input.texCoord;
    op.normal = mul(g_model, float4(input.normal, 0)).xyz;
    return op;
}

PixelInput DefShading_VS(VertexInput input) 
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

//shader x5
float3x3 GetTangentSpaceMatrix3(float3 N, float3 p, float2 uv)
{
    // get edge vectors of the pixel triangle
    float3 dp1 = ddx( p );
    float3 dp2 = ddy( p );
    float2 duv1 = ddx( uv );
    float2 duv2 = ddy( uv );
 
    // solve the linear system
    float3 dp2perp = cross( dp2, N );
    float3 dp1perp = cross( N, dp1 );
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    //if(dot(cross(B, T), N) < 0)
    {
        B *= -1;
    }
 
    // construct a scale-invariant frame 
    float invmax = rsqrt( max( dot(T,T), dot(B,B) ) );
    return transpose(float3x3( T * invmax, B * invmax, N ));
}

float3x3 GetTangentSpaceMatrix3_2(float3 normal, float3 position, float2 texCoord)
{
    float3 dp1 = ddx(position);
    float3 dp2 = ddy(position);
    float2 duv1 = ddx(texCoord);
    float2 duv2 = ddy(texCoord);

    float3x3 M = float3x3(dp1, dp2, cross(dp1, dp2));
    float2x3 inverseM = float2x3(cross(M[1], M[2]), cross(M[2], M[0]));
    float3 T = mul(float2(duv1.x, duv2.x), inverseM);
    float3 B = mul(float2(duv1.y, duv2.y), inverseM);
    return transpose(float3x3(normalize(T), normalize(B), normalize(normal)));
}

PixelOutput DefShading_PS(PixelInput input)
{
    PixelOutput op;
    float4 tex = g_diffuseColor.Sample(g_samplerWrap, g_textureScale * input.texCoords);

    if(tex.a < 0.1) discard;

    //float3 tangent = normalize(input.tangent);
    float3 iNormal = normalize(input.normal);
    //float3 bitangent = cross(tangent, iNormal);
    //tex += g_hasNormalMap.x;
    
    if(g_hasNormalMap.x > 0.5)
    {
        float3x3 nm = GetTangentSpaceMatrix3(iNormal, input.world.xyz, input.texCoords);
        float3 normal = g_normalColor.Sample(g_samplerWrap, g_textureScale * input.texCoords).xyz;
        normal = 2 * normal - float3(1, 1, 1);
        normal = normalize(normal);
        normal = mul(nm, normal);
        op.normal = float4(normal, 0);
    }
    else
    {
       op.normal = float4(normalize(input.normal), 0);
    }

    op.normal.w = dot(normalize(g_CSMlightPos.xyz), iNormal); //peter panning hack sucks balls

    tex = tex * tex;//pow(abs(tex), 2.2);// * tex;//pow(tex, 1.0 / 2.0);// * tex; //gamma (2.0) correction 

   // tex = op.normal;

    op.worldPosDepth = input.world;
    op.worldPosDepth.w = length(input.worldView.xyz);

    op.diffMaterialSpecR = half4(g_diffuseMaterial.x, g_diffuseMaterial.y, g_diffuseMaterial.z, g_specularMaterial.x);
    op.ambientMaterialSpecG = half4(g_ambientMaterial.x, g_ambientMaterial.y, g_ambientMaterial.z, g_specularMaterial.y);
    op.diffuseColorSpecB = half4(tex.x, tex.y, tex.z, g_specularMaterial.z);

    /*op.normal = float4(normalize(input.normal), 0);
    op.diffuseColorSpecB = half4(op.normal.x, op.normal.y, op.normal.z, g_specularMaterial.z); */
    
    op.reflectionStr = (half)g_reflectanceMaterial;

    //HIGHLIGHT_PICKED

    return op;
}

PixelOutput DefShadingWire_PS(PixelInput input)
{
    PixelOutput op;
    if(input.texCoords.x > 0.1 || input.texCoords.x > 0.9)
    {
        //discard;
    }

    float4 tex = g_diffuseColor.Sample(g_samplerWrap, g_textureScale * input.texCoords);
    
    op.worldPosDepth.xyz = input.position.xyz;
    op.worldPosDepth.w = input.position.w;

    op.normal = float4(0,1,0,0);

    op.diffMaterialSpecR = half4(g_diffuseMaterial.x, g_diffuseMaterial.y, g_diffuseMaterial.z, g_specularMaterial.x);
    op.ambientMaterialSpecG = half4(g_ambientMaterial.x, g_ambientMaterial.y, g_ambientMaterial.z, g_specularMaterial.y);
    op.diffuseColorSpecB = half4(tex.x, tex.y, tex.z, g_specularMaterial.z);

    op.reflectionStr = (half)g_reflectanceMaterial;

    return op;
}

PixelOutput DefEditor_PS(PixelInput input)
{
    PixelOutput op;
    
    op.worldPosDepth.xyz = input.position.xyz;
    op.worldPosDepth.w = input.position.w;

    op.normal = float4(input.normal,0);
    op.diffMaterialSpecR = half4(1,1,1,0);
    op.ambientMaterialSpecG = half4(.1,.1,.1,0);
    op.diffuseColorSpecB = half4(.5,.1,.1,0);

    op.reflectionStr = g_selectedActorId.x == g_actorId.x;

    HIGHLIGHT_PICKED

    return op;
}
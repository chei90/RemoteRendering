Texture2D shaderTexture : register(t0);
TextureCube cuebMap : register(t1);

SamplerState SampleType : register(s0);
SamplerState g_samCube : register(s1);

cbuffer MatrixBuffer : register(b0) 
{
    matrix view;
    float3 g_eyePos;
    float unused0;
    float4 unsued1;
    float4 unsued2;
    float4 unsued3;
};

cbuffer MatrixBuffer : register(b1) 
{
    matrix projection;
};

cbuffer ModelMatrix : register(b2) 
{
    matrix model;
};

cbuffer MaterialBuffer : register(b3) 
{
    float4 ambientMaterial;
    float4 specularMaterial;
    float4 diffuseMaterial;
    float specularExpoMaterial;
    float reflectanceMaterial;
    float textureScale;
    float unused;
}

struct PixelInput 
{
    float2 texCoords : TEXCOORD0;
    float4 position : SV_POSITION;
    float4 world : POSITION1;
    float3 normal : NORMAL0;
};

struct PixelOutput 
{
    float4 color : SV_Target0;
};

struct VertexInput {
    float3 position : POSITION0;
    float3 normal : NORMAL0;
    float2 texCoord : TEXCOORD0;
};

static const float4 g_lightDir = float4(0,1,0,0);

static const float3 g_PointLightPos = float3(10,10,0);

PixelInput mainVS(VertexInput input) 
{
    PixelInput op;
    op.world = mul(model, float4(input.position, 1));
    op.world /= op.world.w;
    op.position = mul(projection, mul(view, op.world));
    op.texCoords = input.texCoord;
    op.normal = mul(model, float4(input.normal, 0)).xyz;
    return op;
}

static const float3 weights[4] = {

    float3(1,0,0),
    float3(0,1,0),
    float3(0,0,1),
    float3(1,1,1),
};

PixelOutput mainPS(PixelInput input)
{
    PixelOutput op;
    
    float4 tex = shaderTexture.Sample(SampleType, textureScale * input.texCoords);
    if(tex.a < 0.1) discard;
    float3 lightToPos = normalize(input.world.xyz - g_PointLightPos);
    float3 posToEye = normalize(g_eyePos.xyz - input.world.xyz);
    float3 normal = normalize(input.normal);
    float3 reflectVec = reflect(lightToPos, normal);
    
    //float3 reflectCubeVec = reflect(-posToEye, normal);

    float shadowSample = cuebMap.Sample(g_samCube, lightToPos).r;

    float diffuse = saturate(dot(-lightToPos, normal));

    float specular = pow(saturate(dot(reflectVec, posToEye)), 12);

    float3 pos2Light = g_PointLightPos - input.world.xyz;
    float intensity = 300.0 / (1+dot(pos2Light, pos2Light));

    op.color = diffuseMaterial * diffuse * tex;
    float3 lightToPosShadow = input.world.xyz - g_PointLightPos;
    float distSquared = dot(lightToPosShadow, lightToPosShadow);

    int shadow = shadowSample < (distSquared - 0.05) ? 1 : 0; 
    float fShadow = shadow ? 0.1 : 1;

    op.color += (1-shadow) * specular * specularMaterial;

    op.color *= intensity * fShadow;
    op.color += ambientMaterial * tex;

    return op;
}
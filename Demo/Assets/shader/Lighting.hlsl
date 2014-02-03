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

struct PixelLight
{
    float diffuse;
    float specular;
};

PixelLight GetLightConComponents(float3 posToEye, float3 lightToPos, float3 normal, float specScale)
{
    float3 reflectVec = reflect(lightToPos, normal);

    PixelLight pl;

    pl.diffuse = saturate(dot(-lightToPos, normal));

    pl.specular = pow(saturate(dot(reflectVec, posToEye)), specScale);
    
    return pl;
}

PixelLight GetLightConComponents(float3 eye, float3 world, float3 lightPos, float3 normal, float specScale)
{
    float3 posToEye = normalize(eye - world);

    float3 lightToPos = normalize(world - lightPos);

    float3 reflectVec = reflect(lightToPos, normal);

    PixelLight pl;

    pl.diffuse = saturate(dot(-lightToPos, normal));

    pl.specular = pow(saturate(dot(reflectVec, posToEye)), specScale);
    
    return pl;
}

PixelInput Lighting_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

int isOut(in float3 world, uint index, out float2 tc, out float4 worldInLight)
{
    worldInLight = mul(g_lightProjection[index], mul(g_lightView, float4(world, 1)));
    worldInLight /= worldInLight.w;
    tc = 0.5 * float2(worldInLight.x, worldInLight.y) + 0.5;
    tc = float2(tc.x, 1 - tc.y);
    float nearClip = 0;
    return tc.x < 0 || tc.x > 1 || tc.y < 0 || tc.y > 1 || worldInLight.z < nearClip || worldInLight.z > 1;
}

void computeCSMContr(out in float4 color, in float2 texCoords, in float3 normal)
{

    float3 world = g_worldPosDepth.Sample(g_samplerClamp, texCoords).xyz;
    float lightPosDepth = mul(g_lightView, float4(world, 1)).z;

    uint index = 0;
    float2 tc = float2(0,0);
    float4 worldInLight = float4(0,0,0,0);

    if(!isOut(world, 0, tc, worldInLight))
    {
        index = 0;
    }
    else if(!isOut(world, 1, tc, worldInLight))
    {
        index = 1;
    }
    else if(!isOut(world, 2, tc, worldInLight))
    {
        index = 2;
    }
    else
    {
        return;
    }

    //float4 colorMask = 4 * float4(index == 0, index == 1, index == 2, 1);
    //color *= colorMask;

    float4 gs[3];
    gs[0] = g_effectSource0.Sample(g_samplerClamp, tc);
    gs[1] = g_effectSource1.Sample(g_samplerClamp, tc);
    gs[2] = g_effectSource2.Sample(g_samplerClamp, tc);

    //gs[3] = g_cascadedShadows[3].Sample(g_samplerClamp, tc).x;
    //color = float4(gs[index],0,0,1);

    float2 moments = gs[index].xy;
      
    float dist = g_distances[index];
        
    float rescaled_dist_to_light = (120 + lightPosDepth) / dist;

    float light_shadow_bias = 0;//0.1;//-0.1f;
    
    float light_vsm_epsilon = 0.000001f;
    
    rescaled_dist_to_light -= light_shadow_bias;
    
    float lit_factor = (rescaled_dist_to_light <= moments.x);
    
    // Variance shadow mapping
    float E_x2 = moments.y;
    float Ex_2 = moments.x * moments.x;
    float variance = min(max(E_x2 - Ex_2, 0.0) + light_vsm_epsilon, 1);
    float m_d = (rescaled_dist_to_light - moments.x);
    float p = variance / (variance + m_d * m_d);

    color *= max(lit_factor, max(p, 0.1));
    
    /*if(lightPosDepth > (moments.x + bias))
       {

           // if(dot(dd, float3(0,0,1)) > 0)
            {
                color *= 0.3;//float4(0,0,0,0);
            }
        } */
}

PixelOutput DebugGlobalLighting_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(0,0,0,0);

    float3 normal = 0;
    float3 diffuse = 0;

    float4 nn = g_normals.Sample(g_samplerClamp, input.texCoord);
    normal = nn.xyz;
    diffuse = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord).xyz;

    float3 sunposition = normalize(g_CSMlightPos.xyz);
    diffuse = float3(1,1,1);
    op.color = float4(diffuse * saturate(dot(sunposition, normalize(normal))),1);
    
    return op;
}

PixelOutput GlobalLighting_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(0,0,0,0);

    float3 normal = 0;
    float4 ambientMat = 0;
    float4 diffuseMat = 0;
    float4 diffuse = 0;
    float3 specularMat = 0;

    ambientMat = g_ambientMaterialSpecG.Sample(g_samplerClamp, input.texCoord);
    diffuseMat = g_diffuseMaterialSpecR.Sample(g_samplerClamp, input.texCoord);
    float4 nn = g_normals.Sample(g_samplerClamp, input.texCoord);
    normal = nn.xyz;
    diffuse = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord);
    specularMat = float3(diffuseMat.w, ambientMat.w, diffuse.w);

    float3 sunposition = normalize(g_CSMlightPos.xyz);

    float3 sunIntensity = float3(g_lightIntensity.x, g_lightIntensity.y, g_lightIntensity.z); //todo

    float4 worldDepth = g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);
    float d = worldDepth.w;

    float4 sun = 0.9 * float4(253 / 255.0, 184 / 255.0, 0 / 255.0, 0);
    float4 sky = 0;//0.7 * float4(135 / 255.0, 206 / 255.0, 1, 0);

    if(d > 0)
    {
        //reflectance
        float ref = g_normalColor.Sample(g_samplerClamp, input.texCoord).x;
        float3 skyTex = float3(0,0,0);
        if(ref > 0)
        {
            float3 reflectVec = normalize(reflect(worldDepth.xyz - g_eyePos.xyz, normal));
            float u = 0.5 + atan2(reflectVec.z, reflectVec.x) / (2 * PI);
            float v = 0.5 - asin(reflectVec.y) / PI;
            float2 tc = float2(u,1-1.5*v);
            skyTex = g_diffuseColor.Sample(g_samplerClamp, tc).xyz;
            skyTex *= skyTex;
            float refScale = saturate(0.6 + dot(float3(0,1,0), reflectVec));
            skyTex *= refScale;
        }

        int selfShade = nn.w < 0;

        diffuse = float4(lerp(diffuse.xyz, skyTex, ref), 0);
        //op.color = float4(normal, 1);

        PixelLight pl = GetLightConComponents(normalize(g_eyePos.xyz - worldDepth.xyz), -sunposition, normalize(normal), 8);

        op.color = float4(sunIntensity * (diffuseMat.xyz * diffuse.xyz * pl.diffuse + specularMat * pl.specular), 1);

        //op.color += float4(ambientMat.xyz * diffuse.xyz, 0);
        //CSM
        /*if(selfShade) //hack to avoid peter panning, todo
        {
            op.color *= (0.65 + 0.4 * nn.w);
        }
        else*/
        {
            computeCSMContr(op.color, input.texCoord, normal);
        }

        op.color += g_ambient * diffuse * diffuseMat; //float4(g_ambient.xyz * ambientMat.xyz, 0); // * diffuse.xyz
    } 
    else
    {
        float4 ray = float4(-1.0 +  2.0 * float2(input.texCoord.x, 1 - input.texCoord.y), 1, 0);
        ray.y *= 1;
        ray.w = 0;
        ray = mul(g_invView, ray);
        ray = normalize(ray);

        float4 tex = diffuse;

        float powa = pow(saturate(dot(ray.xyz, sunposition)), 32);
        
        float l = clamp(worldDepth.y * 0.1, 0, 1);
        op.color = 0.80000005*diffuse;
    }
    return op;
}

PixelOutput PointLighting_PS(PixelInput input)
{
    PixelOutput op;

    float3 normal = 0;
    float3 ambientMat = 0;
    float3 diffuseMat = 0;
    float3 specularMat = 0;
    float3 diffuseColor = 0;
    float3 world = 0;
    float depth = 0;

    float4 wd = g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);
    world = wd.xyz; 
    depth = wd.w;

    /*if(depth == 0)
    {
        discard;
    } */

    float3 lightPos = g_lightPos.xyz;

    float3 d = world - lightPos;

    float radius = g_lightColorRadius.w;

    float distSquared = dot(d, d);

    if(distSquared > radius*radius || (depth <= 0)) 
    {
        discard;
    }

    float3 lightToPos = normalize(world - lightPos);

    float shadowSample = g_pointLightShadowMap.Sample(g_samplerClamp, lightToPos).r;

    normal = g_normals.Sample(g_samplerClamp, input.texCoord).xyz;

    int hasNormal = abs(dot(normal, normal)) > 0;

    if(hasNormal)
    {
        normal = normalize(normal);
    }

    float4 dmsmr = g_diffuseMaterialSpecR.Sample(g_samplerClamp, input.texCoord);
    float4 amsmg = g_ambientMaterialSpecG.Sample(g_samplerClamp, input.texCoord);
    float4 dcsmb = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord);

    ambientMat = amsmg.xyz;
    diffuseMat = dmsmr.xyz;
    specularMat = float3(dmsmr.w, amsmg.w, dcsmb.w);

    diffuseColor = dcsmb.xyz;

    float3 lightColor = g_lightColorRadius.xyz;

    float intensity = max(0, 1 - distSquared / (radius * radius));

    float bias = 0.15;//0.005 * tan(acos(dot(-lightToPos, normal)));//0.15
    //bias = saturate(bias);
    int shadow = shadowSample < (distSquared - bias) ? 1 : 0;

    if(shadow && hasNormal)
    {
        discard;
    }

    if(!hasNormal)
    {
        op.color = float4(intensity * lightColor * diffuseColor, 1);
        return op;
    }

    float3 posToEye = normalize(g_eyePos.xyz - world);

    float3 reflectVec = reflect(lightToPos, normal);

    //float bias = (distSquared * DepthBias) - shadowSample;

    float s = sign(dot(abs(normal), abs(normal)));

    float diffuse = s * saturate(dot(-lightToPos, normal)) + (1-s);

    float specular = s * pow(saturate(dot(reflectVec, posToEye)), 32) + (1-s);

    float3 color = 0;

    color = lightColor * (specular * specularMat + diffuseMat * diffuse * diffuseColor);
    
    float intensityScale = g_lightPos.w;
    intensity *= intensityScale;

    color *= intensity; // * fShadow;

    op.color = float4(color, 1);

    return op;
}

PixelOutput SpotLighting_PS(PixelInput input)
{
    PixelOutput op;

    float3 normal = 0;
    float3 ambientMat = 0;
    float3 diffuseMat = 0;
    float3 specularMat = 0;
    float3 diffuseColor = 0;
    float3 world = 0;
    float depth = 0;

    float4 wd =  g_worldPosDepth.Sample(g_samplerClamp, input.texCoord);
    world = wd.xyz; 
    depth = wd.w;

    /*if(depth == 0)
    {
        discard;
    } */

    float3 lightPos = g_lightPos.xyz;

    float3 d = world - lightPos;

    float length = g_lightColorRadius.w;

    float distSquared = dot(d, d);

    float3 dir = normalize(g_lightViewDirAngle.xyz);
    float angle = g_lightViewDirAngle.w;

    if(distSquared > length*length) 
    {
        discard;
    } 
    
    float3 lightToPos = normalize(world - lightPos);
    float lpd = max(dot(lightToPos, dir), 0);
    float a = acos(lpd);

    if(a > angle * 0.5)
    {
        discard;
    }
    
    float4 worldLight = mul(g_projection, mul(g_view, float4(world, 1)));
    worldLight /= worldLight.w;
    float2 tc = 0.5 * float2(worldLight.x, worldLight.y) + 0.5;
    tc = float2(tc.x, 1-tc.y);

    float shadowSample = g_diffuseColor.Sample(g_samplerClamp, tc).r;
    float bias = 0.15;

    int shadow = g_castShadow[0] * (shadowSample < (distSquared - bias) ? 1 : 0);

    normal = g_normals.Sample(g_samplerClamp, input.texCoord).xyz;

    int hasNormal = abs(dot(normal, normal)) > 0;

    if(hasNormal)
    {
        normal = normalize(normal);
    }

    /*if(shadow && hasNormal)
    {
        discard;
    }*/

    float4 dmsmr = g_diffuseMaterialSpecR.Sample(g_samplerClamp, input.texCoord);
    float4 amsmg = g_ambientMaterialSpecG.Sample(g_samplerClamp, input.texCoord);
    float4 dcsmb = g_diffuseColorSpecB.Sample(g_samplerClamp, input.texCoord);

    ambientMat = amsmg.xyz;
    diffuseMat = dmsmr.xyz;
    specularMat = float3(dmsmr.w, amsmg.w, dcsmb.w);
    diffuseColor = dcsmb.xyz;

    float intensity = (angle * 0.5 - a) * (1  - distSquared / (length * length));
    float3 lightColor = g_lightColorRadius.xyz;

    /*if(!hasNormal)
    {
        op.color = float4(intensity * lightColor * diffuseColor, 1);
        return op;
    }*/

    float3 posToEye = normalize(g_eyePos.xyz - world);

    float3 reflectVec = reflect(lightToPos, normal);

    //float bias = (distSquared * DepthBias) - shadowSample;

    //float s = sign(dot(abs(normal), abs(normal)));

    float diffuse = saturate(dot(-lightToPos, normal));// + (1-s);

    float specular = pow(saturate(dot(reflectVec, posToEye)), 32);// + (1-s);

    intensity = intensity < 0 ? 0 : intensity;
    float intensityScale = g_lightPos.w;
    intensity *= intensityScale;
    
    float3 projTex = g_normalColor.Sample(g_samplerClamp, float2(tc.x, 1 - tc.y)).rgb;

    float3 color = projTex * lightColor * (specular * specularMat + diffuseMat * diffuse * diffuseColor);
    
    color *= intensity;

    op.color = !hasNormal * float4(intensity * lightColor * diffuseColor, 1) 
        + hasNormal * !shadow * float4(color, 1);

    return op;
}



struct PixelInput 
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
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


PixelInput RT_VS(VertexInput input) 
{
    PixelInput op;
    op.position = float4(input.position, 1);
    op.texCoord = input.texCoord;
    return op;
}

PixelOutput RT_PS(PixelInput input)
{
    PixelOutput op;
    op.color = float4(input.texCoord, 0, 0);
    return op;
}
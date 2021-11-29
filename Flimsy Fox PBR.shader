// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'

Shader "Flimsy Fox/PBR 1.0.1 Transparent"
{
    Properties
    {
		[HideInInspector] shader_is_using_thry_editor ("", Float) = 0
		[HideInInspector] shader_master_label ("<color=#00ff00ff>Flimsy Fox PBR 1.0.1</color>", Float) = 0
		[HideInInspector] shader_properties_label_file("FFPBRPoiLabels", Float) = 0

		[HideInInspector] footer_github ("github footer button", Float) = 0
		
		[HideInInspector] m_mainOptions("Shader Settings", Float) = 0
		_NumSamples ("Number of samples", Range(1, 1024)) = 1
		[Toggle(_)]_EnableRefl ("Reflections Toggle", Float) = 1
		
		[HideInInspector]m_start_Albedo("Albedo", Float) = 0
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB/RGBA)", 2D) = "white" {}
		[HideInInspector]m_end_Albedo("Albedo", Float) = 0
		
		[HideInInspector]m_start_Specular("Specular", Float) = 0
        _Specular ("Specular (BW)", 2D) = "(1,1,1,1)" {}
		_SpecularAdd ("Add Specular", Range(0.0, 1.0)) = 0
		[HideInInspector]m_end_Specular("Specular", Float) = 0
		
		[HideInInspector]m_start_Roughness("Roughness", Float) = 0
        _Smoothness ("Roughness (BW)", 2D) = "(1,1,1,1)" {}
		_RoughnessMult ("Multiplier", Range(0.0, 1.0)) = 1
		[HideInInspector]m_end_Roughness("Roughness", Float) = 0
		
		[HideInInspector]m_start_Normals("Normals", Float) = 0
		_BumpMap ("Normal", 2D) = "(1,1,1,1)" {}
		[Toggle(_)] _MakeDisplacement ("Displacement map", Float) = 0
		_DisplacementMult ("Distance (mm)", Float) = 0
		[HideInInspector]m_end_Normals("Normals", Float) = 0
		
		[HideInInspector]m_start_Emission("Emission", Float) = 0
		_EmissionColor ("Emission Color", Color) = (1,1,1,1)
		_Emission ("Emission (RGB)", 2D) = "none" {}
		_EmissionMask ("Emission Mask (BW)", 2D) = "(1,1,1,1)" {}
		_EmissionStrength ("Emission Strength", Float) = 1
		
		[HideInInspector]m_start_Glow("Glow in the Dark", Float) = 0
		[Toggle(_)] _GlowInTheDarkEnable ("Glow in the dark", Float) = 0
		_GlowInTheDarkMax ("Glow in the dark max light", Float) = 0.25
		[HideInInspector]m_end_Glow("Glow in the Dark", Float) = 0
		[HideInInspector]m_end_Emission("Emission", Float) = 0
		
		[HideInInspector]m_start_AudioLink("AudioLink", Float) = 0
		[HideInInspector]_AudioLink ("AudioLink Texture", 2D) = "black" {}
		[Toggle(_)]_AudioLinkEnable ("Enable AudioLink", Float) = 0
		[Enum(Local, 0, UV, 1)] _AudioLinkSpace("Coordinate Space", Float) = 0
		_Height ("Height (Meters)", Float) = 2
		_AudioLinkKey ("AudioLink Color Key", Color) = (0.5,0.5,0.5,1)
		_AudioLinkKeyRange ("AudioLink Key Range", Range(0.0, 1.0)) = 0.5
		[HideInInspector]m_end_AudioLink("AudioLink", Float) = 0
    }
	
	CustomEditor "ThryEditor"
    SubShader
    {
		Pass
		{
			Tags {"LightMode"="ForwardBase"}
			LOD 350

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#pragma glsl
			#pragma target 3.0
			#include "UnityCG.cginc"
			#include "Assets/AudioLink/Shaders/AudioLink.cginc"
			#include "UnityLightingCommon.cginc"
			
			static const float PI = 3.14159265f;
			float test = 232e-9;
			
			float _Height;
			float _NumSamples;
			int _EnableRefl;
			
			fixed4 _Color;
			sampler2D _MainTex;
			
			sampler2D _Specular;
			float _SpecularAdd;
			
			sampler2D _Smoothness;
			float _RoughnessMult;
			
			sampler2D _BumpMap;
			int _MakeDisplacement;
			float _DisplacementMult;
			
			fixed4 _EmissionColor;
			sampler2D _Emission;
			sampler2D _EmissionMask;
			float _EmissionStrength;
			
			int _GlowInTheDarkEnable;
			float _GlowInTheDarkMax;
			
			int _AudioLinkEnable;
			int _AudioLinkSpace;
			float4 _AudioLinkKey;
			float _AudioLinkKeyRange;
			
			float3 uNormal;
			
			struct appdata
			{
				float4 vertex : POSITION;
				
				float4 tangent : TANGENT;
				float3 normal : NORMAL;
				
				float2 uv : TEXCOORD0;
			};
			
			struct v2f
			{
				float3 worldPos : TEXCOORD0;
				
				float2 uv : TEXCOORD20;
				half3 tspace0 : TEXCOORD30; 
                half3 tspace1 : TEXCOORD40; 
                half3 tspace2 : TEXCOORD50;
				float3 worldViewDir : TEXCOORD60;
				float3 normal : NORMAL;				
				
				float4 vertex : POSITION;
			};
			
			float clampLoop(float input, float max)
			{
				return input % max;
			}
			
			bool testRange(float f, float mid, float ran)
			{
				if(f >= mid - ran && f <= mid + ran)
					return true;
				else
					return false;
			}
			
			float energy(float3 color)
			{
				return dot(color, 1.0f / 3.0f);
			}
			
			float rand(float2 uv)
			{
				return frac(sin(dot(uv,float2(12.9898,78.233)))*43758.5453123);
			}
			
			float sdot(float3 x, float3 y, float f = 1.0f)
			{
				return saturate(dot(x,y) * f);
			}
			
			float3x3 GetTangentSpace(float3 normal)
			{
				// Choose a helper vector for cross product
				float3 helper = float3(1, 0, 0);
				if(abs(normal.x) > 0.99f)
					helper = float3(0, 0, 1);
				
				//Generate vectors
				float3 tangent = normalize(cross(normal, helper));
				float3 binormal = normalize(cross(normal, tangent));
				return float3x3(tangent, binormal, normal);
			}
			
			float3 SampleHemisphere(float2 uv, float3 normal, float alpha)
			{
				//Sample hemisphere, where alpha determines kind of sampling
				float cosTheta = pow(rand(uv), 1.0f / (alpha + 1.0f));
				float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
				float phi = 2 * PI * rand(uv);
				float3 tangentSpaceDir = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
				
				// Transform direction to world space
				return mul(tangentSpaceDir, GetTangentSpace(normal));
			}
			
			float SmoothnessToPhongAlpha(float s)
			{
				return pow(1000.0f, s * s);
			}
			
			v2f vert(appdata v)
			{
				v2f o;
				o.uv = v.uv;
				
                // world space normal
                float3 worldNormal = UnityObjectToWorldNormal(v.normal);
				half3 wTangent = UnityObjectToWorldDir(v.tangent.xyz);
                // compute bitangent from cross product of normal and tangent
                half tangentSign = v.tangent.w * unity_WorldTransformParams.w;
                half3 wBitangent = cross(worldNormal, wTangent) * tangentSign;
                // output the tangent space matrix
                o.tspace0 = half3(wTangent.x, wBitangent.x, worldNormal.x);
                o.tspace1 = half3(wTangent.y, wBitangent.y, worldNormal.y);
                o.tspace2 = half3(wTangent.z, wBitangent.z, worldNormal.z);
				
				#if !defined(SHADER_API_OPENGL)
				if(_MakeDisplacement)
				{
					fixed3 normal = fixed3(0,0,0);
					normal = UnpackNormal(tex2Dlod(_BumpMap, float4(v.uv, 0, 0)));
					v.vertex.x += (normal.x) * _DisplacementMult/1000;
					v.vertex.y += (normal.y) * _DisplacementMult/1000;
					v.vertex.z += (normal.z) * _DisplacementMult/1000;
					
					float3 posPlusTangent = v.vertex + v.tangent * 0.01;
					float3 bitangent = cross(v.normal, v.tangent);
					float3 posPlusBitangent = v.vertex + bitangent * 0.01;
					float3 modifiedTangent = posPlusTangent - v.vertex;
					float3 modifiedBitangent = posPlusBitangent - v.vertex;
					float3 modifiedNormal = cross(modifiedTangent, modifiedBitangent);
					o.normal = normalize(modifiedNormal);
				}
				#endif
				
				o.vertex = UnityObjectToClipPos(v.vertex);
				
				o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                // compute world space view direction
                o.worldViewDir = normalize(UnityWorldSpaceViewDir(o.worldPos));
				return o;
			}
			
			fixed4 frag(v2f IN) : COLOR
			{
				float4 origAlbedo;
				float4 emission;
				float4 emissionMask;
				float4 audioLink;
				float3 uPos;
				float4 origin = unity_ObjectToWorld[3];
				
				float LIGHTMULT = 2;
			
				//Calculate world normals
				if(!_MakeDisplacement)
				{
					half3 worldNormal;
					half3 normal = UnpackNormal(tex2D(_BumpMap, IN.uv));
					worldNormal.x = dot(IN.tspace0, normal);
					worldNormal.y = dot(IN.tspace1, normal);
					worldNormal.z = dot(IN.tspace2, normal);
					uNormal = worldNormal;
				}
				else
				{
					uNormal = mul( unity_ObjectToWorld, float4( IN.normal, 0.0 ) ).xyz;
				}
				
				float4 worldRefl = float4(reflect(-IN.worldViewDir, uNormal), 1);
				
				//Reflected light, color, and shadow calculations
				origAlbedo = float4(tex2D (_MainTex, IN.uv) * _Color);
				float4 reflectionColor = float4(0,0,0,0);
				float4 lightMapColor;
				if(_EnableRefl == 1)
				{
					reflectionColor = float4(UNITY_SAMPLE_TEXCUBE (unity_SpecCube0, uNormal));
					reflectionColor = float4(DecodeHDR(half4(reflectionColor), unity_SpecCube0_HDR), reflectionColor.w);
				}
				lightMapColor = float4(_LightColor0.rgb, 1)/LIGHTMULT;
				lightMapColor += float4(ShadeSH9(half4(uNormal,1)), 0)/LIGHTMULT;
				reflectionColor += lightMapColor;
				reflectionColor = min(reflectionColor, 1);
				origAlbedo *= lightMapColor;
				
				float4 specular = 1 - min(float4(tex2D (_Specular, IN.uv)) + _SpecularAdd, 1);
				float4 smoothness = (float4(tex2D (_Smoothness, IN.uv))) * _RoughnessMult;
				emissionMask = float4(tex2D (_EmissionMask, IN.uv));
				emission = float4(tex2D (_Emission, IN.uv));
				emission.a *= emissionMask.r;
				if(_AudioLinkEnable &&
					testRange(emission.r, _AudioLinkKey.r, _AudioLinkKeyRange) &&
					testRange(emission.g, _AudioLinkKey.g, _AudioLinkKeyRange) &&
					testRange(emission.b, _AudioLinkKey.b, _AudioLinkKeyRange))
				{
					if(_AudioLinkSpace == 0)
						uPos = IN.worldPos - origin;
					else if(_AudioLinkSpace == 1)
						uPos = IN.uv.x * IN.uv.y;
					audioLink = AudioLinkData( ALPASS_AUDIOLINK + uint2( 0, (IN.worldPos - IN.vertex).y/_Height * 4. ) ).rrrr;
					//audioLink = float4(audioLink.w, audioLink.z, audioLink.y, audioLink.x);
					emission.rgba *= audioLink;
				}
				emission *= _EmissionColor * _EmissionStrength;
				
				float4 finalAlbedo = float4(0,0,0,0);
				float2 pos = IN.uv.xy;
				
				//PBR shading starts
				float4 albedo = min(1.0f - specular, reflectionColor);
				float specChance = energy(specular);
				float diffChance = energy(albedo);
				float sum = specChance + diffChance;
				specChance /= sum;
				diffChance /= sum;
				
				for(int i = 0; i < _NumSamples; i++)
				{
					float roulette = rand(pos.x + pos.y + 2 * i/_NumSamples);
					if(roulette < specChance)
					{
						//Specular
						float alpha = SmoothnessToPhongAlpha((1 - smoothness));
						float3 direction = SampleHemisphere(pos, uNormal, alpha);
						float f = (alpha + 2) / (alpha + 1);
						finalAlbedo += (origAlbedo + fixed4(origAlbedo * (1.0f / specChance) * 
							specular * sdot(uNormal, direction, f)))/_NumSamples;
					}
					else
					{
						//Diffuse
						finalAlbedo += (fixed4(origAlbedo *(1.0f / diffChance) * albedo))/_NumSamples;
					}
				}
				float glowInTheDark;
				if(_GlowInTheDarkEnable)
					glowInTheDark = 1 - min(lightMapColor + (1 - _GlowInTheDarkMax/LIGHTMULT), 1);
				else
					glowInTheDark = 1;
				finalAlbedo += emission * glowInTheDark;
				/*finalAlbedo = float4(clampLoop(uPos.y/_Height, 1)
					, clampLoop(uPos.y/_Height, 1)
					,  clampLoop(uPos.y/_Height, 1), 1);*/
				//finalAlbedo = float4(IN.normal, 0);
				return fixed4(finalAlbedo);
			}
			ENDCG
		}
		Pass
        {
            Tags {"LightMode"="ShadowCaster"}

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster
            #include "UnityCG.cginc"

            struct v2f { 
                V2F_SHADOW_CASTER;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }
	}
	//Fallback "Diffuse"
}
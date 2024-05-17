Shader "Flimsy Fox/PBR 1.3.0a Opaque"
{
    Properties
    {
		[HideInInspector] shader_is_using_thry_editor ("", Float) = 0
		[HideInInspector] shader_master_label ("<color=#00ff00ff>Flimsy Fox PBR 1.3.0a</color>", Float) = 0
		[HideInInspector] shader_properties_label_file("FFOXLabels", Float) = 0

		[HideInInspector] footer_github ("github footer button", Float) = 0
		
		[HideInInspector] m_mainOptions("Shader Settings", Float) = 0
		_NumSamples ("Number of samples", Range(1, 256)) = 128
		[HideInInspector]_LightMult ("Lighting Multiplier", Range(0, 5)) = 1
		[HideInInspector][Toggle(_)]_EnableRefl ("Reflections Toggle", Float) = 1
		
		[HideInInspector]m_start_Albedo("Albedo", Float) = 0
        _Color ("Color", Color) = (1,1,1,1)
        [NoScaleOffset] _Albedo ("Albedo (RGB/RGBA)", 2D) = "white" {}
		[HideInInspector]m_end_Albedo("Albedo", Float) = 0
		
		[HideInInspector]m_start_Specular("Specular", Float) = 0
        [NoScaleOffset] _Specular ("Specular (BW)", 2D) = "(1,1,1,1)" {}
		_SpecularMult("Multiply", Range(0.0, 1.0)) = 1
		_SpecularAdd ("Add", Range(0.0, 1.0)) = 0
		[HideInInspector]m_end_Specular("Specular", Float) = 0
		
		[HideInInspector]m_start_Roughness("Roughness", Float) = 0
        [NoScaleOffset] _Roughness ("Roughness (BW)", 2D) = "(1,1,1,1)" {}
		[Toggle(_)] _SmoothnessToggle ("As Smoothness Map", Float) = 0
		_RoughnessMult ("Multiply", Range(0.0, 1.0)) = 1
		_RoughnessAdd ("Add", Range(0.0, 1.0)) = 0
		[HideInInspector]m_end_Roughness("Roughness", Float) = 0
		
		[HideInInspector]m_start_Normals("Normals", Float) = 0
		[Toggle(_)] _EnableBumpMap ("Enable Normal", Float) = 0
		[NoScaleOffset] _BumpMap ("Normal", 2D) = "(0,0,0,1)" {}
		[Toggle(_)] _EnableNormal1 ("Enable Normal 2", Float) = 0
		[NoScaleOffset] _Normal1 ("Normal 2", 2D) = "(0,0,0,1)" {}
		[HideInInspector]m_start_Height("Height Map", Float) = 0
		[Toggle(_)] _EnableDisplacement ("Enable Displacement", Float) = 0
		_DisplacementMult ("Distance (mm)", Float) = 0
		[NoScaleOffset] _HeightMap ("Height Map", 2D) = "(0.5,0.5,0.5,1)" {}
		[HideInInspector]m_end_Height("Height Map", Float) = 0
		[HideInInspector]m_end_Normals("Normals", Float) = 0
		
		[HideInInspector]m_start_Emission("Emission", Float) = 0
		_EmissionColor ("Emission Color", Color) = (1,1,1,1)
		[NoScaleOffset] _Emission ("Emission (RGB)", 2D) = "none" {}
		[NoScaleOffset] _EmissionMask ("Emission Mask (BW)", 2D) = "(1,1,1,1)" {}
		_EmissionStrength ("Emission Strength", Float) = 0
		
		[HideInInspector]m_start_Glow("Glow in the Dark", Float) = 0
		[Toggle(_)] _GlowInTheDarkEnable ("Glow in the dark", Float) = 0
		_GlowInTheDarkMax ("Glow in the dark max light", Float) = 0.25
		[HideInInspector]m_end_Glow("Glow in the Dark", Float) = 0
		[HideInInspector]m_end_Emission("Emission", Float) = 0
		
		[HideInInspector]m_start_AudioLink("AudioLink", Float) = 0
		[HideInInspector]_AudioLink ("AudioLink Texture", 2D) = "black" {}
		[Toggle(_)]_AudioLinkEnable ("Enable AudioLink", Float) = 0
		[HideInInspector]m_start_coordSpace("Coordinate Settings", Float) = 0
		[Enum(None, 0, Local, 1, UV, 3)] _AudioLinkSpace("Coordinate Space", Float) = 0
		[Toggle(_)]_InvertALCoord("Invert", Float) = 0
		[HideInInspector]m_end_coordSpace("Coordinate Settings", Float) = 0
		_Height ("Height (Meters)", Float) = 2
		[Enum(Bass, 0, LowMid, 1, LowHigh, 2, Treble, 3)] _ALBand("Audio Band", Float) = 0
		[HideInInspector]m_start_AL_colorkey("Color Key", Float) = 0
		_AudioLinkKey ("AudioLink Color Key", Color) = (0.5,0.5,0.5,1)
		_AudioLinkKeyRange ("AudioLink Key Range", Range(0.0, 1.0)) = 0.5
		[HideInInspector]m_end_AL_colorkey("Color Key", Float) = 0
		[HideInInspector]m_end_AudioLink("AudioLink", Float) = 0

		[HideInInspector]m_start_Fallback("Fallback", Float) = 0
		[NoScaleOffset] _MainTex ("Texture", 2D) = "Black" {}
		[NoScaleOffset] _OcclusionMap("Occlusion", 2D) = "white" {}
		[HideInInspector]m_end_Fallback("Fallback", Float) = 0
    }
	
	CustomEditor "Thry.ShaderEditor"
    SubShader
    {
		Pass
		{
			Tags {"LightMode"="ForwardBase" 
			"RenderType"="Transparent"
			}
			LOD 350
			ZWrite On
			Blend SrcAlpha OneMinusSrcAlpha

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#pragma glsl
			#pragma target 3.0
			#pragma shader_feature _EMISSION
			#pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "UnityStandardCore.cginc"
			#include "AutoLight.cginc"
			#include "UnityStandardUtils.cginc"
			#include "Assets/Flimsy Fox/Shaders/common/audio-link/Shaders/AudioLink.cginc"
			
			static const float PI = 3.14159265f;
			float test = 232e-9;
			float _Seed = 124;
			float2 _Pixel = float2(0,0);
			float3 _WorldPos = float3(123,314,532);
			
			float _Height;
			float _NumSamples;
			float _LightMult;
			float _UberVolumetricMode;
			int _EnableRefl;
			
			//fixed4 _Color;
			sampler2D _Albedo;
			float4 _Albedo_ST;
			
			sampler2D _Specular;
			float _SpecularMult;
			float _SpecularAdd;
			
			sampler2D _Roughness;
			float _SmoothnessToggle;
			float _RoughnessMult;
			float _RoughnessAdd;
			
			int _EnableBumpMap;
			//sampler2D _BumpMap;
			int _EnableNormal1;
			sampler2D _Normal1;
			int _EnableDisplacement;
			float _DisplacementMult;
			sampler2D _HeightMap;
			
			//fixed4 _EmissionColor;
			sampler2D _Emission;
			sampler2D _EmissionMask;
			float _EmissionStrength;
			
			int _GlowInTheDarkEnable;
			float _GlowInTheDarkMax;
			
			int _AudioLinkEnable;
			int _AudioLinkSpace;
			int _InvertALCoord;
			int _ALBand;
			float4 _AudioLinkKey;
			float _AudioLinkKeyRange;
			
			float3 uNormal;
			float3 normalTest;
			
			struct appdata
			{
				float4 vertex : POSITION;
				
				float4 tangent : TANGENT;
				float3 normal : NORMAL;
				
				float2 uv : TEXCOORD0;
				float2 texcoord1 : TEXCOORD1;
			};
			
			struct vertexOutput
			{
				float3 worldPos : TEXCOORD0;
				float3 localPos : TEXCOORD1;
				float4 screenPos : TEXCOORD5;
				SHADOW_COORDS(10)
				UNITY_FOG_COORDS(15)
				float2 uv : TEXCOORD20;
				half4 ambientoruvLM : TEXCOORD10;
				float4 tangent : TANGENT;
				half3 tspace0 : TEXCOORD30; 
                half3 tspace1 : TEXCOORD40; 
                half3 tspace2 : TEXCOORD50;
				float3 worldViewDir : TEXCOORD60;
				float3 normal : NORMAL;				
				
				float4 vertex : POSITION;
			};

			struct PBRLight
			{
				float size;
				float3 position;
				float3 intensity;
			};
			
			float clampLoop(float input, float max)
			{
				return abs(input) % max;
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
			
			float rand()
			{
				float3 x = float3(_Seed, _Pixel);
				
				float result = frac(sin(x.x / 100.0f *
					dot(x.yz, float2(12.9898f, 78.233f))) *
					43758.5453f);
				
				float4 xx = float4(_Seed, abs(_WorldPos)*1000000000);
				
				uint UI0 = 1597334673U;
				uint UI1 = 3812015801U;
				uint3 UI3 = uint3(UI0, UI1, 2798796415U);
				uint4 UI4 = uint4(UI3, 1985387995U);
				float UIF = (1.0 / float(0xffffffffU));
				
				
				uint4 q = uint4(xx) * UI4;
				q *= UI4;
				uint n = (q.w ^ q.x ^ q.y ^ q.z) * UI0;
				//result = float(n) * UIF;
				
				_Seed += 1;
				
				return result;
			}

			float4x4 fastInverseMatrix4x4(float4x4 inputMatrix)
			{
				//https://www.codeproject.com/Questions/754429/C-Program-to-calculate-inverse-of-matrix-n-n
				float4x4 inverse_matrix;
				float det=determinant(inputMatrix);

				float num=1/det;
				float4x4 m_Transpose=transpose(inputMatrix);

				/*complex of determinant with Transpose*/
				for(int i=0;i<4;i++)
				{
					for(int j=0;j<4;j++)
					{
						inverse_matrix[i][j]=num*m_Transpose[i][j];
					}
				}


				return inverse_matrix;
			}
			
			float sdot(float3 x, float3 y, float f = 1.0f)
			{
				return saturate(dot(x,y) * f);
			}
			
			float3 ConstructNormal(float3 v1, float3 v2, float3 v3)
			{
				return normalize(cross(v2 - v1, v3 - v1));
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
			
			float3 SampleHemisphere(float3 v, float3 normal, float alpha)
			{
				//Redefine variables for easy copy-paste ;P
				float3 Ve = v;
				float alpha_x = alpha;
				float alpha_y = alpha;
				float U1 = rand();
				float U2 = rand();
				float3 Nh = normal;
				
				//Thanks to this scientific paper for helping me out with this code to the return line:
				//https://jcgt.org/published/0007/04/01/paper.pdf
				
				//Calculations
				// Section 3.2: transforming the view direction to the hemisphere configuration
				float3 Vh = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
				// Section 4.1: orthonormal basis (with special case if cross product is zero)
				float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
				float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1,0,0);
				float3 T2 = cross(Vh, T1);
				// Section 4.2: parameterization of the projected area
				float r = sqrt(U1);
				float phi = 2.0 * PI * U2;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + Vh.z);
				t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
				// Section 4.3: reprojection onto hemisphere
				Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
				// Section 3.4: transforming the normal back to the ellipsoid configuration
				float3 Ne = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
				return Ne;
			}

			float2 sphereIntersect( in float3 ro, in float3 rd, in float3 ce, float ra )
			{
				float3 oc = ro - ce;
				float b = dot( oc, rd );
				float3 qc = oc - b*rd;
				float h = ra*ra - dot( qc, qc );
				if( h<0.0 ) return float2(-1.0, -1.0); // no intersection
				h = sqrt( h );
				return float2( -b-h, -b+h );
			}
			
			float SmoothnessToPhongAlpha(float s)
			{
				return pow(1000.0f, s * s);
			}

			float3 shadeDiffuse(PBRLight light, inout float3 lighting, float3 worldPosition, float3 direction, float diffChance, float3 albedo)
			{
				float3 intensity = 0;
				float2 intersect = sphereIntersect(worldPosition, direction, light.position, light.size);
				if(intersect.y >= 0)
				{
					intensity = (light.intensity * (1.0f / diffChance) *
						albedo);
					lighting += light.intensity;
				}

				return intensity;
			}

			float3 shadeSpecular(PBRLight light, inout float3 lighting, float3 worldPosition, float3 normal, float3 direction
				, float f, float specChance, float3 albedo, float3 specular)
			{
				float3 intensity = 0;
				float2 intersect = sphereIntersect(worldPosition, direction, light.position, light.size);
				if(intersect.y >= 0)
				{
					intensity = (light.intensity * (1.0f / specChance) * 
						specular * sdot(normal, direction, f));
					lighting += light.intensity;
				}
				return intensity;
			}

			float3 traceAndShade(float screenSize, PBRLight lights[7], half3 lightmap, inout float3 lighting
				, float3 worldPosition, float3 normal, float3 viewDirection
				, float3 albedo, float3 specular, float3 smoothness)
			{
				float3 intensity = 0;
				float roulette = rand();
				albedo = min(1.0f - specular, albedo);
				float specChance = energy(specular);
				float diffChance = energy(albedo);
				float sum = specChance + diffChance;
				specChance /= sum;
				diffChance /= sum;
				
				float alpha = SmoothnessToPhongAlpha(smoothness);
				float3 direction = SampleHemisphere(viewDirection, normal, alpha);
				float f = (alpha + 2) / (alpha + 1);
				if(roulette < specChance)
				{
					//Lightmap
					half3 ambient;
					half4 lightmapUV;
					float4 lightmapColor = float4(0,0,0,0);
					#if defined(LIGHTMAP_ON) || defined(DYNAMICLIGHTMAP_ON)
						ambient = 0;
						lightmapUV = lightmap;
					#else
						ambient = lightmap.rgb;
						lightmapUV = 0;
					#endif
					#if defined(LIGHTMAP_ON)
						half4 bakedColorTex = UNITY_SAMPLE_TEX2D(unity_Lightmap, lightmapUV.xy);
						half3 bakedColor = DecodeLightmap(bakedColorTex);
						#ifdef DIRLIGHTMAP_COMBINED
							fixed4 bakedDirTex = UNITY_SAMPLE_TEX2D_SAMPLER(unity_LightmapInd, unity_Lightmap, lightmapUV.xy);
							lightmapColor += DecodeDirectionalLightmap(bakedColor, bakedDirTex, normal);
						#else
							lightmapColor += bakedColor;
						#endif
					#endif
					#ifdef DYNAMICLIGHTMAP_ON
						fixed4 realtimeColorTex = UNITY_SAMPLE_TEX2D(unity_DynamicLightmap, lightmapUV.zw);
						half3 realtimeColor = DecodeRealtimeLightmap(realtimeColorTex);
						#ifdef DIRLIGHTMAP_COMBINED
							half4 realtimeDirTex = UNITY_SAMPLE_TEX2D_SAMPLER(unity_DynamicDirectionality, unity_DynamicLightmap, lightmapUV.zw);
							lightmapColor += DecodeDirectionalLightmap(realtimeColor, realtimeDirTex, normal);
						#else
							lightmapColor += realtimeColor;
						#endif
					#endif
					lights[4].intensity = lightmapColor.rgb;
					lights[4].position = worldPosition + normal;
					lights[4].size = screenSize;

					//Cubemap
					float4 reflectionColor = float4(0,0,0,1);
					reflectionColor = UNITY_SAMPLE_TEXCUBE (unity_SpecCube0, normal);
					reflectionColor = float4(DecodeHDR(half4(reflectionColor), unity_SpecCube0_HDR), reflectionColor.w);
					lights[5].intensity = reflectionColor;
					lights[5].position = worldPosition + normal; //INVESTIGATE: is there a better way to get CubeMap position in a PBR manner?
					lights[5].size = screenSize;

					//Ambient lighting, if no lightmap
					lights[6].intensity = ambient;
					lights[6].position = worldPosition;
					lights[6].size = screenSize;

					for(int i = 0; i < 7; i++)
					{
						intensity += shadeSpecular(lights[i], lighting, worldPosition, normal, direction
							, f, specChance, albedo, specular);
					}
				}
				//Diffuse
				else
				{
					//Lightmap
					half3 ambient;
					half4 lightmapUV;
					float4 lightmapColor = float4(0,0,0,0);
					#if defined(LIGHTMAP_ON) || defined(DYNAMICLIGHTMAP_ON)
						ambient = 0;
						lightmapUV = lightmap;
					#else
						ambient = lightmap.rgb;
						lightmapUV = 0;
					#endif
					#if defined(LIGHTMAP_ON)
						half4 bakedColorTex = UNITY_SAMPLE_TEX2D(unity_Lightmap, lightmapUV.xy);
						half3 bakedColor = DecodeLightmap(bakedColorTex);
						#ifdef DIRLIGHTMAP_COMBINED
							fixed4 bakedDirTex = UNITY_SAMPLE_TEX2D_SAMPLER(unity_LightmapInd, unity_Lightmap, lightmapUV.xy);
							lightmapColor += DecodeDirectionalLightmap(bakedColor, bakedDirTex, direction);
						#else
							lightmapColor += bakedColor;
						#endif
					#endif
					#ifdef DYNAMICLIGHTMAP_ON
						fixed4 realtimeColorTex = UNITY_SAMPLE_TEX2D(unity_DynamicLightmap, lightmapUV.zw);
						half3 realtimeColor = DecodeRealtimeLightmap(realtimeColorTex);
						#ifdef DIRLIGHTMAP_COMBINED
							half4 realtimeDirTex = UNITY_SAMPLE_TEX2D_SAMPLER(unity_DynamicDirectionality, unity_DynamicLightmap, lightmapUV.zw);
							lightmapColor += DecodeDirectionalLightmap(realtimeColor, realtimeDirTex, direction);
						#else
							lightmapColor += realtimeColor;
						#endif
					#endif
					lights[4].intensity = lightmapColor.rgb;
					lights[4].position = worldPosition + direction;
					lights[4].size = screenSize;

					//Cubemap
					float4 reflectionColor = float4(0,0,0,1);
					reflectionColor = UNITY_SAMPLE_TEXCUBE (unity_SpecCube0, direction);
					reflectionColor = float4(DecodeHDR(half4(reflectionColor), unity_SpecCube0_HDR), reflectionColor.w);
					lights[5].intensity = reflectionColor;
					lights[5].position = worldPosition + direction; //INVESTIGATE: is there a better way to get CubeMap distance in a PBR manner?
					lights[5].size = screenSize;

					//Ambient lighting, if no lightmap
					lights[6].intensity = ambient;
					lights[6].position = worldPosition;
					lights[6].size = screenSize;

					for(int i = 0; i < 7; i++)
					{
						intensity += shadeDiffuse(lights[i], lighting, worldPosition, direction, diffChance, albedo);
					}
				}
				return intensity;
			}
			vertexOutput vert(appdata v)
			{
				vertexOutput o;
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

				fixed3 heightMap = tex2Dlod(_HeightMap, float4(v.uv, 0, 0));
				v.vertex += (float4(v.normal, 0) - .5) * _DisplacementMult/1000 * _EnableDisplacement;
				
				o.worldPos = mul(unity_ObjectToWorld, v.vertex);
				o.localPos = v.vertex;
				o.worldViewDir = (NormalizePerVertexNormal(_WorldSpaceCameraPos - o.worldPos.yxz) + 2)/2 * float3(1,1,1);

				o.tangent = v.tangent;
				o.normal = v.normal;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				UNITY_TRANSFER_FOG(o,o.vertex);

				VertexInput unityAppdata;
				unityAppdata.vertex = v.vertex;
				unityAppdata.normal = v.normal;
				unityAppdata.uv0 = v.uv;
				float3 vertexWorldNormal = UnityObjectToWorldNormal(v.normal);
				o.ambientoruvLM = VertexGIForward(unityAppdata, o.worldPos, vertexWorldNormal);
				
				TRANSFER_SHADOW(o)
				return o;
			}
			
			fixed4 frag (vertexOutput IN) : COLOR
			{
				float4 albedo;
				float4 emission;
				float4 emissionMask;
				float4 audioLink;
				float3 uPos;
				float4 origin = unity_ObjectToWorld[3];
				
				float2 screenUV = IN.screenPos.xy / IN.screenPos.w;
				_Pixel = screenUV * _ScreenParams.xy;
				float3 viewDirection = (normalize(_WorldSpaceCameraPos - IN.worldPos.yxz) + 2)/2;
				
				//Normals
				half3 baseNormal = UnpackNormal(float4(0.5,0.5,1,1));
				half3 normal0 = lerp(baseNormal, UnpackNormal(tex2D(_BumpMap, IN.uv)), _EnableBumpMap);
				half3 normal1 = lerp(baseNormal, UnpackNormal(tex2D(_Normal1, IN.uv)), _EnableNormal1);
				half3 normal = normalize(baseNormal + normal0 + normal1);
				
				uNormal.x = dot(IN.tspace0, normal);
				uNormal.y = dot(IN.tspace1, normal);
				uNormal.z = dot(IN.tspace2, normal);
				
				albedo = tex2D (_Albedo, IN.uv) * _Color;
				
				float3 specular = tex2D (_Specular, IN.uv);
				float3 roughness = tex2D (_Roughness, IN.uv);
				roughness *= _RoughnessMult + _RoughnessAdd;
				float3 smoothness = lerp(roughness, 1 - roughness, _SmoothnessToggle);
				
				specular = min(specular * _SpecularMult + _SpecularAdd, 1);
				
				emissionMask = float4(tex2D (_EmissionMask, IN.uv));
				emission = float4(tex2D (_Emission, IN.uv));
				emission *= _EmissionColor * _EmissionStrength;

				//AudioLink
				if(_AudioLinkEnable &&
					testRange(emission.r, _AudioLinkKey.r, _AudioLinkKeyRange) &&
					testRange(emission.g, _AudioLinkKey.g, _AudioLinkKeyRange) &&
					testRange(emission.b, _AudioLinkKey.b, _AudioLinkKeyRange))
				{
					uPos = IN.worldPos - origin;
					audioLink = AudioLinkData( ALPASS_AUDIOLINK + uint2(3 - _ALBand, 0)).rrrr;

					switch(_AudioLinkSpace)
					{
					case(0):
					{
						emission.rgba *= audioLink;
						break;
					}
					case(1):
					{
						emission.rgba *= lerp((-IN.localPos.y + _Height/2)/_Height, 
								1 - (-IN.localPos.y + _Height/2)/_Height, _InvertALCoord) < audioLink;
						break;
					}
					}
				}
				
				//PBR shading starts
				float3 colorOut = float3(0,0,0);
				float3 lighting = 0;
				
				//Hit 1
				for(int i = 0; i < _NumSamples; i++)
				{

					//Get lighting info for first-bounce ray casting
					PBRLight lights[7]; //0-3 PointLights; 4 lightmap; 5 cubemap; 6 ambient

					//Point Lights
					for (int index = 0; index < 4; index++)
					{  
						lights[index].position = float3(unity_4LightPosX0[index], 
						unity_4LightPosY0[index], 
						unity_4LightPosZ0[index]);    //TODO: fast inverse matrix
						//lights[index].position = mul(fastInverseMatrix4x4(UNITY_MATRIX_MV), lights[index].position).xyz;
						lights[index].intensity = unity_LightColor[index].rgb;
						lights[index].size = 1; //TODO: get actual light size

					}
					
					colorOut += traceAndShade(IN.screenPos.w, lights, IN.ambientoruvLM, lighting
				, IN.worldPos, uNormal, viewDirection
				, albedo, specular, smoothness);
				}
				colorOut /= _NumSamples;
				lighting /= _NumSamples;
				
				float3 glowInTheDark = 1;
				if(_GlowInTheDarkEnable)
				{
					glowInTheDark *= max(min(lighting - _GlowInTheDarkMax, 1), 0);
				}
				emission.r *= emissionMask.r;
				emission.g *= emissionMask.g;
				emission.b *= emissionMask.b;
				
			 	colorOut += emission * emission.a * glowInTheDark;
				
				//POST PROCESSING and final calculations
				UNITY_APPLY_FOG(IN.fogCoord, colorOut);
			 	//colorOut.a = origAlbedo.a;
				//colorOut = emission;
				
				return fixed4(colorOut, albedo.a);
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
}
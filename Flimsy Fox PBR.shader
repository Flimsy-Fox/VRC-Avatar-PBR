Shader "Flimsy Fox/PBR 1.3.0a Opaque"
{
    Properties
    {
		[HideInInspector] shader_is_using_thry_editor ("", Float) = 0
		[HideInInspector] shader_master_label ("<color=#00ff00ff>Flimsy Fox PBR 1.3.0a</color>", Float) = 0
		[HideInInspector] shader_properties_label_file("FFOXLabels", Float) = 0

		[HideInInspector] footer_github ("github footer button", Float) = 0
		
		_deltaTime ("deltaTime", Float) = 0.033
		
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
		
		[HideInInspector]m_start_Post("Post Processing", Float) = 0
		[Toggle(_)]_enableDenoise("Enable Denoising", Float) = 0
		[HideInInspector]_denoiseTexture ("Denoising Texture", 2D) = "black" {}
		[Toggle(_)]_enableAGX("Enable AGX", Float) = 1
		[HideInInspector]m_end_Post("Post Processing", Float) = 0

		[HideInInspector]m_start_Debug("Debug", Float) = 0
		[HideInInspector]m_start_Frag_Debug("Fragment Shader", Float) = 0
		[Enum(none, 0, albedo, 0.2, emission, 0.4, normal, 0.6, lighting, 0.8, shading, 1)] _FragDebugMode("Debug Mode", Float) = 100
		[Enum(point 0, 0, point 1, 0.167, point 2, 0.333, point 3, 0.5, lightmap, 0.667, cubemap, 0.833, ambient, 1)] _FragDebugLightIndex ("Light Index", Range(0, 6)) = 0
		[Enum(shadeNormal, 0, shadeNormalDiff, 0.5, shadeEmission, 1)] _FragDebugShadeMode ("Shade Mode", Float) = 0
		[HideInInspector]m_end_Frag_Debug("Fragment Shader", Float) = 0
		[HideInInspector]m_end_Debug("Debug", Float) = 0

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

			#pragma exclude_renderers d3d11_9x
			#pragma exclude_renderers d3d9
			#pragma vertex vert
			#pragma fragment frag
			#pragma shader_feature _EMISSION
			#pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "UnityStandardCore.cginc"
			#include "AutoLight.cginc"
			#include "UnityStandardUtils.cginc"
			#include "Assets/Flimsy Fox/Shaders/common/audio-link/Shaders/AudioLink.cginc"
			
			static const float PI = 3.14159265f;
			static const float numPointLights = 4;
			static const float numOtherLights = 3; // lightmap, cubemap, ambient
			static const float numTotalLights = numPointLights + numOtherLights;
			static const float frametimeTarget = 0.0111;
			float _Seed = 124;
			float2 _Pixel = float2(0,0);
			float3 _WorldPos = float3(123,314,532);
			
			float _Height;
			float _deltaTime;
			//float _UberVolumetricMode;
			
			//TODO: Move _Color, _BumpMap, and _EmissionColor into Fallback section; already defined by include files
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

			int _enableDenoise;
			sampler2D _denoiseTexture;
			int _enableAGX;

			float _FragDebugMode;
			float _FragDebugLightIndex;
			float _FragDebugShadeMode;
			
			float3 uNormal;
			
			struct appdata
			{
				float4 vertex : POSITION;
				
				float4 tangent : TANGENT;
				float3 normal : NORMAL;
				
				float2 uv : TEXCOORD0;
				float2 texcoord1 : TEXCOORD1;
			};
			
			struct VertexOutput
			{
				float3 worldPos : TEXCOORD0;
				float3 localPos : TEXCOORD1;
				float4 screenPos : TEXCOORD5;
				float4 vertex : POSITION;
				float2 uv : TEXCOORD20;
				float4 tangent : TANGENT;

				half3x3 tspace : TEXCOORD30;
				half4 ambientoruvLM : TEXCOORD10;

				SHADOW_COORDS(10)
				UNITY_FOG_COORDS(15)
			};

			struct PBRLight
			{
				float size;
				float3 position;
				float3 intensity;
			};

			struct FragDebug
			{
				half3 albedo;
				half3 emission;
				half3 normal;
				
				half3 lightingColor[numTotalLights];
				half3 shadeNormal;
				half3 shadeNormalDiff;
				half3 shadeEmission;
			};
			
			FragDebug fragDebugConstruct(float3 albedo, float3 emission, float3 normal)
			{
				FragDebug fragDebug = (FragDebug)0;
				fragDebug.albedo = albedo;
				fragDebug.emission = emission;
				fragDebug.normal = normal;
				return fragDebug;
			}

			FragDebug fragDebugPost(FragDebug fragDebug, float3 emission, int sampleCount)
			{
				for(int i = 0; i < numTotalLights; i++)
				{
					fragDebug.lightingColor[i] /= sampleCount;
				}
				fragDebug.shadeNormal /= sampleCount;
				fragDebug.shadeNormalDiff = ((fragDebug.shadeNormal-fragDebug.normal)+1)/2;
				if(0.45 < fragDebug.shadeNormalDiff.r > 0.55)
				{
					fragDebug.shadeNormalDiff.r = 0.5;
				}
				if(0.45 < fragDebug.shadeNormalDiff.g > 0.55)
				{
					fragDebug.shadeNormalDiff.g = 0.5;
				}
				if(0.45 < fragDebug.shadeNormalDiff.b > 0.55)
				{
					fragDebug.shadeNormalDiff.b = 0.5;
				}
				fragDebug.shadeEmission = emission;
				return fragDebug;
			}
			float3 displayDebug(float3 colorOut, FragDebug fragDebug)
			{
				int fragDebugMode = int(_FragDebugMode*5);
				int fragDebugLightIndex = int(_FragDebugLightIndex*numTotalLights-1);
				int fragDebugShadeMode = int(_FragDebugShadeMode*2);

				switch(fragDebugMode)
				{
				case(0):
				{
					break;
				}
				case(1):
				{
					colorOut = fragDebug.albedo;
					break;
				}
				case(2):
				{
					colorOut = fragDebug.emission;
					break;
				}
				case(3):
				{
					colorOut = fragDebug.normal;
					break;
				}
				case(4):
				{
					colorOut = fragDebug.lightingColor[fragDebugLightIndex];
					break;
				}
				case(5):
				{
					switch(fragDebugShadeMode)
					{
					case(0):
					{
						colorOut = fragDebug.shadeNormal;
						break;
					}
					case(1):
					{
						colorOut = fragDebug.shadeNormalDiff;
						break;
					}
					case(2):
					{
						colorOut = fragDebug.shadeEmission.rgb;
						break;
					}
					}
					break;
				}
				}
				return colorOut;
			}
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
				result *= float(n) * UIF;
				
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

			float3 rotateVector(float3 vec, float3 anglesRadian)
			{
				//https://stackoverflow.com/questions/14607640/rotating-a-vector-in-3d-space
				//Rotate about the X-axis
				vec.y = vec.y*cos(anglesRadian.x)-vec.z*sin(anglesRadian.x);
				vec.z = vec.y*sin(anglesRadian.x)+vec.z*cos(anglesRadian.x);
				
				//Rotate about the Y-axis
				vec.x = vec.x*cos(anglesRadian.y)+vec.z*sin(anglesRadian.y);
				vec.z = -vec.x*sin(anglesRadian.y)+vec.z*cos(anglesRadian.y);

				//Rotate about the Z-axis
				vec.x = vec.x*cos(anglesRadian.z)-vec.y*sin(anglesRadian.z);
				vec.y = vec.x*sin(anglesRadian.z)+vec.y*cos(anglesRadian.z);

				return vec;
			}

			float vectorAngle2(float2 vec) {
				//https://stackoverflow.com/a/48227232
				float rad90 = PI/2;
				float rad180 = PI;
				float rad270 = 2*PI*(3/4);
				if (vec.x == 0) // special cases
					return (vec.y > 0)? rad90
						: (vec.y == 0)? 0
						: rad270;
				else if (vec.y == 0) // special cases
					return (vec.x >= 0)? 0
						: rad180;
				float ret = atan(vec.y/vec.x);
				if (vec.x < 0 && vec.y < 0) // quadrant Ⅲ
					ret = rad180 + ret;
				else if (vec.x < 0) // quadrant Ⅱ
					ret = rad180 + ret; // it actually substracts
				else if (vec.y < 0) // quadrant Ⅳ
					ret = rad270 + (rad90 + ret); // it actually substracts
				return ret;
			}

			//TODO: To sphere or to hemisphere?
			float3 sampleSphere(float3 viewDirection, float3 normal, float3 alpha)
			{
				// Define PI if not already defined
				#ifndef PI
				#define PI 3.141592653589793
				#endif

				// Generate orthogonal tangent and bitangent vectors
				float3 tangent;
				
				// Handle vertical normals to avoid cross product issues
				if (abs(normal.y) > 0.999)
					tangent = normalize(cross(normal, float3(1, 0, 0)));
				else
					tangent = normalize(cross(normal, float3(0, 1, 0)));
				
				float3 bitangent = normalize(cross(normal, tangent));

				// Generate random angles scaled by weights
				float angleX = (2 * rand() - 1) * PI * alpha.x;
				float angleY = (2 * rand() - 1) * PI * alpha.y;
				float angleZ = (2 * rand() - 1) * PI * alpha.z;

				// Create individual rotation matrices
				float sinX = sin(angleX);
				float cosX = cos(angleX);
				float3x3 rotX = float3x3(
					1, 0, 0,
					0, cosX, -sinX,
					0, sinX, cosX
				);

				float sinY = sin(angleY);
				float cosY = cos(angleY);
				float3x3 rotY = float3x3(
					cosY, 0, sinY,
					0, 1, 0,
					-sinY, 0, cosY
				);

				float sinZ = sin(angleZ);
				float cosZ = cos(angleZ);
				float3x3 rotZ = float3x3(
					cosZ, -sinZ, 0,
					sinZ, cosZ, 0,
					0, 0, 1
				);

				// Combine rotations in ZYX order
				float3x3 rotation = mul(rotZ, mul(rotY, rotX));

				// Create transformation matrices
				float3x3 localToWorld = float3x3(tangent, bitangent, normal);
				float3x3 worldToLocal = transpose(localToWorld);

				// Transform vector to tangent space
				float3 localViewDirection = mul(worldToLocal, viewDirection);
				
				// Apply combined rotation
				float3 rotatedLocal = mul(rotation, localViewDirection);
				
				// Transform back to world space
				return mul(localToWorld, rotatedLocal);
			}

			float2 sphereIntersect( in float3 ro, in float3 rd, in float3 ce, float ra )
			{
				//https://iquilezles.org/articles/intersectors/
				float3 oc = ro - ce;
				float b = dot( oc, rd );
				float3 qc = oc - b*rd;
				float h = ra*ra - dot( qc, qc );
				if( h<0.0 ) return float2(-1.0, -1.0); // no intersection
				h = sqrt( h );
				return float2( -b-h, -b+h );
			}
			
			float smoothnessToAlpha(float s)
			{
				return 1*s*s*s*s;
			}

			float3 shadeDiffuse(PBRLight light, inout float3 lighting, float3 worldPosition, float3 direction, 
				float diffChance, float3 albedo)
			{
				float3 intensity = 0;
				float2 intersect = sphereIntersect(worldPosition, direction, light.position, light.size);
				if(intersect.y >= 0)
				{
					intensity = (light.intensity * (1.0f / diffChance) *
						albedo);
				}

				return intensity;
			}

			float3 shadeSpecular(PBRLight light, inout float3 lighting, float3 worldPosition, float3 normal, float3 direction
				, float f, float specChance, float3 specular)
			{
				float3 intensity = 0;
				float2 intersect = sphereIntersect(worldPosition, direction, light.position, light.size);
				if(intersect.y >= 0)
				{
					intensity = (light.intensity * (1.0f / specChance) * 
						specular * sdot(normal, direction, f));
				}
				return intensity;
			}

			float3 traceAndShade(float screenSize, half3 lightmap, inout float3 lighting
				, float3 worldPosition, float3 normal, float3 viewDirection
				, float3 albedo, float3 specular, float3 alpha, inout FragDebug fragDebug)
			{
				//float3 alpha = smoothnessToAlpha(smoothness);
				//float3 alpha = smoothness;
				//alpha.z = 0;

				//TODO: Debug using Debug normal mode
				float3 direction = sampleSphere(-viewDirection, normal, alpha);
				
				PBRLight lights[numPointLights+numOtherLights];

				//Point Lights
				for (int index = 0; index < numPointLights; index++)
				{  
					lights[index].position = float3(unity_4LightPosX0[index], 
					unity_4LightPosY0[index], 
					unity_4LightPosZ0[index]);    //TODO: fast inverse matrix
					//lights[index].position = mul(unity_ObjectToWorld, lights[index].position).xyz;
					lights[index].intensity = unity_LightColor[index].rgb;
					lights[index].size = (0.005 * sqrt(1000000.0 - unity_4LightAtten0.x)) / sqrt(unity_4LightAtten0.x);

				}

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
				lights[numPointLights].intensity = lightmapColor.rgb;
				lights[numPointLights].position = worldPosition + direction;
				lights[numPointLights].size = screenSize;

				//Cubemap
				float4 reflectionColor = float4(0,0,0,1);
				reflectionColor = UNITY_SAMPLE_TEXCUBE (unity_SpecCube0, direction);
				reflectionColor = float4(DecodeHDR(half4(reflectionColor), unity_SpecCube0_HDR), reflectionColor.w);
				lights[numPointLights+1].intensity = reflectionColor;
				lights[numPointLights+1].position = worldPosition + direction; //INVESTIGATE: is there a better way to get CubeMap distance in a PBR manner?
				lights[numPointLights+1].size = screenSize;

				//Ambient lighting, if no lightmap
				lights[numPointLights+2].intensity = ambient;
				lights[numPointLights+2].position = worldPosition;
				lights[numPointLights+2].size = screenSize;
				
				float3 intensity = 0;
				float roulette = rand();
				albedo = min(1.0f - specular, albedo);
				float specChance = energy(specular);
				float diffChance = energy(albedo);
				float sum = specChance + diffChance;
				specChance /= sum;
				diffChance /= sum;

				//direction = normal;
				float f = (energy(alpha) + 2) / (energy(alpha) + 1);
				if(roulette < specChance)
				{
					for(int i = 0; i < numTotalLights; i++)
					{
						float3 lightIntensity = shadeSpecular(lights[i], lighting, worldPosition, normal, 
							direction, f, specChance, specular);
						intensity += lightIntensity;
						lighting += lightIntensity;
						fragDebug.lightingColor[i] += lightIntensity;
						fragDebug.shadeNormal += normal;
					}
					//intensity += -(direction - normal);
				}
				//Diffuse
				else
				{

					for(int i = 0; i < numTotalLights; i++)
					{
						float3 lightIntensity = shadeDiffuse(lights[i], lighting, worldPosition, direction, 
							diffChance, albedo);
						intensity += lightIntensity;
						lighting += lightIntensity;
						fragDebug.lightingColor[i] += lightIntensity;
						fragDebug.shadeNormal += direction;
					}
				}

				return intensity;
			}


			//Thanks to https://www.shadertoy.com/view/cd3XWr for the code!
			float3 AGXDefaultContrastApprox(float3 x)
			{
				float3 x2 = x * x;
				float3 x4 = x2 * x2;
				
				return + 15.5     * x4 * x2
						- 40.14    * x4 * x
						+ 31.96    * x4
						- 6.868    * x2 * x
						+ 0.4298   * x2
						+ 0.1191   * x
						- 0.00232;
			}

			float3 AGX(float3 color)
			{
				const float3x3 agx_mat = float3x3(
					0.842479062253094, 0.0423282422610123, 0.0423756549057051,
					0.0784335999999992,  0.878468636469772,  0.0784336,
					0.0792237451477643, 0.0791661274605434, 0.879142973793104);
					
				const float min_ev = -12.47393f;
				const float max_ev = 4.026069f;

				// Input transform
				color = mul(agx_mat, color);
				
				// Log2 space encoding
				color = clamp(log2(color), min_ev, max_ev);
				color = (color - min_ev) / (max_ev - min_ev);
				
				// Apply sigmoid function approximation
				color = AGXDefaultContrastApprox(color);

				return color;
			}

			float3 AGXEotf(float3 color)
			{
				const float3x3 agx_mat_inv = float3x3(
					1.19687900512017, -0.0528968517574562, -0.0529716355144438,
					-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
					-0.0990297440797205, -0.0989611768448433, 1.15107367264116);
					
				// Undo input transform
				color = mul(agx_mat_inv, color);
				
				// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
				color = pow(color, 2.2);

				return color;
			}

			float3 AGXLook(float3 color)
			{
				const float3 lw = float3(0.2126, 0.7152, 0.0722);
				float luma = mul(color, lw);
				
				// Default
				float3 offset = float3(0,0,0);
				float3 slope = float3(1,1,1);
				float3 power = float3(1,1,1);
				float sat = 1.0;
				
				// ASC CDL
				color = pow(color * slope + offset, power);
				return luma + sat * (color - luma);
			}

			float3 AGXTransform(float3 color)
			{
				color = AGX(color);
				color = AGXLook(color);
				color = AGXEotf(color);
				return color;
			}

			float denoiseStrength = 3.0f;

			float3 denoise(float2 denoiseUV, sampler2D denoiseTexture) 
			{
				int2 offset[25];
				offset[0] = int2(-2,-2);
				offset[1] = int2(-1,-2);
				offset[2] = int2(0,-2);
				offset[3] = int2(1,-2);
				offset[4] = int2(2,-2);
				
				offset[5] = int2(-2,-1);
				offset[6] = int2(-1,-1);
				offset[7] = int2(0,-1);
				offset[8] = int2(1,-1);
				offset[9] = int2(2,-1);
				
				offset[10] = int2(-2,0);
				offset[11] = int2(-1,0);
				offset[12] = int2(0,0);
				offset[13] = int2(1,0);
				offset[14] = int2(2,0);
				
				offset[15] = int2(-2,1);
				offset[16] = int2(-1,1);
				offset[17] = int2(0,1);
				offset[18] = int2(1,1);
				offset[19] = int2(2,1);
				
				offset[20] = int2(-2,2);
				offset[21] = int2(-1,2);
				offset[22] = int2(0,2);
				offset[23] = int2(1,2);
				offset[24] = int2(2,2);
				
				
				float kernel[25];
				kernel[0] = 1.0f/256.0f;
				kernel[1] = 1.0f/64.0f;
				kernel[2] = 3.0f/128.0f;
				kernel[3] = 1.0f/64.0f;
				kernel[4] = 1.0f/256.0f;
				
				kernel[5] = 1.0f/64.0f;
				kernel[6] = 1.0f/16.0f;
				kernel[7] = 3.0f/32.0f;
				kernel[8] = 1.0f/16.0f;
				kernel[9] = 1.0f/64.0f;
				
				kernel[10] = 3.0f/128.0f;
				kernel[11] = 3.0f/32.0f;
				kernel[12] = 9.0f/64.0f;
				kernel[13] = 3.0f/32.0f;
				kernel[14] = 3.0f/128.0f;
				
				kernel[15] = 1.0f/64.0f;
				kernel[16] = 1.0f/16.0f;
				kernel[17] = 3.0f/32.0f;
				kernel[18] = 1.0f/16.0f;
				kernel[19] = 1.0f/64.0f;
				
				kernel[20] = 1.0f/256.0f;
				kernel[21] = 1.0f/64.0f;
				kernel[22] = 3.0f/128.0f;
				kernel[23] = 1.0f/64.0f;
				kernel[24] = 1.0f/256.0f;
				
				float3 sum = float3(0,0,0);
				float c_phi = 1.0;
				float4 cval = tex2D(denoiseTexture, denoiseUV);
				
				float cum_w = 0.0;
				for(int i=0; i<25; i++)
				{
					float2 sampleUV = denoiseUV+offset[i]*denoiseStrength;
					
					float3 ctmp = tex2D(denoiseTexture, sampleUV).rgb * tex2D(denoiseTexture, sampleUV).a;
					float3 t = cval - ctmp;
					float dist2 = dot(t,t);
					float c_w = min(exp(-(dist2)/c_phi), 1.0);
					
					float weight = c_w;
					sum += ctmp*weight*kernel[i];
					cum_w += weight*kernel[i];
				}
				return sum/cum_w;
			}

			VertexOutput vert(appdata v)
			{
				VertexOutput o;
				o.uv = v.uv;
				
                // world space normal
                float3 worldNormal = UnityObjectToWorldNormal(v.normal);
				half3 wTangent = UnityObjectToWorldDir(v.tangent.xyz);
                // compute bitangent from cross product of normal and tangent
                half tangentSign = v.tangent.w * unity_WorldTransformParams.w;
                half3 wBitangent = cross(worldNormal, wTangent) * tangentSign;
                // output the tangent space matrix
                o.tspace[0] = half3(wTangent.x, wBitangent.x, worldNormal.x);
                o.tspace[1] = half3(wTangent.y, wBitangent.y, worldNormal.y);
                o.tspace[2] = half3(wTangent.z, wBitangent.z, worldNormal.z);

				fixed3 heightMap = tex2Dlod(_HeightMap, float4(v.uv, 0, 0));
				v.vertex += (float4(v.normal, 0) - .5) * _DisplacementMult/1000 * _EnableDisplacement;
				
				o.worldPos = mul(unity_ObjectToWorld, v.vertex);
				o.localPos = v.vertex;

				o.tangent = v.tangent;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				UNITY_TRANSFER_FOG(o,o.vertex);

				VertexInput unityAppdata;
				unityAppdata.vertex = v.vertex;
				unityAppdata.normal = v.normal;
				unityAppdata.uv0 = v.uv;
				float3 vertexWorldNormal = UnityObjectToWorldNormal(v.normal);
				o.ambientoruvLM = VertexGIForward(unityAppdata, o.worldPos, vertexWorldNormal);
				if(o.ambientoruvLM.r == 0 &&
					o.ambientoruvLM.g == 0 &&
					o.ambientoruvLM.b == 0)
				{
					o.ambientoruvLM = half4(0.5,0.5,0.5,1);
				}
				
				TRANSFER_SHADOW(o)
				return o;
			}
			
			fixed4 frag (VertexOutput IN) : COLOR
			{
				FragDebug fragDebug;
				if(_deltaTime == 0)
				{
					_deltaTime = 0.35;
				}
				int sampleCount = max(1,(frametimeTarget/_deltaTime)*32);
				float4 albedo;
				float4 emission;
				float4 emissionMask;
				float4 audioLink;
				float3 uPos;
				float4 origin = unity_ObjectToWorld[3];
				
				float2 screenUV = IN.screenPos.xy / IN.screenPos.w;
				_Pixel = screenUV * _ScreenParams.xy;
				float3 viewDirection = normalize(IN.worldPos.xyz - _WorldSpaceCameraPos);
				
				//Normals
				half3 baseNormal = UnpackNormal(float4(0.5,0.5,1,1));
				half3 normal0 = lerp(baseNormal, UnpackNormal(tex2D(_BumpMap, IN.uv)), _EnableBumpMap);
				half3 normal1 = lerp(baseNormal, UnpackNormal(tex2D(_Normal1, IN.uv)), _EnableNormal1);
				half3 normal = normalize(baseNormal + normal0 + normal1);
				
				uNormal.x = dot(IN.tspace[0], normal);
				uNormal.y = dot(IN.tspace[1], normal);
				uNormal.z = dot(IN.tspace[2], normal);
				
				albedo = tex2D (_Albedo, IN.uv) * _Color;
				
				float3 specular = tex2D (_Specular, IN.uv);
				float3 roughness = tex2D (_Roughness, IN.uv);
				roughness *= _RoughnessMult + _RoughnessAdd;
				float3 smoothness = lerp(roughness, 1 - roughness, _SmoothnessToggle);
				//assume smoothness does not define transmission
				smoothness/=2;
				
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
				fragDebug = fragDebugConstruct(albedo, emission, uNormal);

				//PBR shading starts
				float3 colorOut = float3(0,0,0);
				float3 lighting = 0;
				
				//Ray-Tracing
				[loop]
				for(int i = 0; i < sampleCount; i++)
				{
					colorOut += traceAndShade(IN.screenPos.w, IN.ambientoruvLM, lighting
				, IN.worldPos, uNormal, viewDirection
				, albedo, specular, smoothness, fragDebug);
				}
				colorOut /= sampleCount;
				lighting /= sampleCount;
				
				float3 glowInTheDark = 1;
				if(_GlowInTheDarkEnable)
				{
					glowInTheDark *= 1-min(lighting.rgb, _GlowInTheDarkMax)/_GlowInTheDarkMax;
				}
				emission *= emissionMask;
				
			 	colorOut += emission.rgb * emission.a * glowInTheDark;
				
				//POST PROCESSING and final calculations
				if(_enableDenoise)
				{
					colorOut = denoise(screenUV, _denoiseTexture);
				}
				UNITY_APPLY_FOG(IN.fogCoord, colorOut);
				fragDebug = fragDebugPost(fragDebug, colorOut - albedo, sampleCount);
				colorOut = displayDebug(colorOut, fragDebug);
				if(_FragDebugMode < 0.001 && _enableAGX)
				{
					#if defined(UNITY_USE_DEBUG_COLORS) || defined(UNITY_DEBUG_DISPLAY)

					#else
					colorOut = AGXTransform(colorOut);
					#endif
				}
				
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
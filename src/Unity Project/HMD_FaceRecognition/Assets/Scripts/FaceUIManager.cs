// FaceUIManager.cs

using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Valve.VR;

public class FaceUIManager : MonoBehaviour
{
    [Header("UI Element References (From Prefab)")]
    public TextMeshProUGUI nameText;
    public Image backgroundPanel;
    public Canvas nameLabelCanvas;
    public Camera nameLabelCamera;
    public RenderTexture nameLabelRenderTexture;
    
    public int update_each_X_frames = 1;


    private ulong frameOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;
    private ulong nameLabelOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;

    private Texture2D frameTexture;
    private Vector3 anchoredPosition;
    private Quaternion anchoredRotation;
    private string currentFaceName;
    private bool isAnchorSet = false;
    private int frameCounter = 0;

    // --- Public properties set by FaceTrackingManager ---
    public float frameWorldWidth { get; set; }
    public float frameAspectRatio { get; set; }
    public int frameBorderWidth { get; set; }
    public float nameLabelOffset { get; set; }
    public float nameLabelWidth { get; set; }
    public float horizontalFov { get; set; }
    public float verticalFov { get; set; }
    public float distanceFactor { get; set; }
    public float OFFSET { get; set; }

    public void Initialize(Face faceData, Transform hmdTransform, string uniqueKey)
    {
        currentFaceName = faceData.name;
        gameObject.name = $"FaceUI_{uniqueKey}";

        // Create own RenderTexture instance
        RenderTextureDescriptor desc = nameLabelRenderTexture.descriptor;
        var rt = new RenderTexture(desc) {
            name = $"NameLabelRT_{uniqueKey}"
        };
        rt.Create();

        nameLabelRenderTexture = rt;

        nameLabelCamera.targetTexture = rt;

        
        Debug.Log($"[FaceUIManager] Initializing for '{currentFaceName}'. Creating overlays.");

        var frameErr = OpenVR.Overlay.CreateOverlay($"Frame_{uniqueKey}", $"Frame_{uniqueKey}", ref frameOverlayHandle);
        if (frameErr != EVROverlayError.None) { Debug.LogError($"[FaceUIManager] Frame overlay creation failed for {currentFaceName}: {frameErr}"); return; }
        OpenVR.Overlay.SetOverlayWidthInMeters(frameOverlayHandle, frameWorldWidth);
        OpenVR.Overlay.SetOverlayAlpha(frameOverlayHandle, 0f);
        OpenVR.Overlay.ShowOverlay(frameOverlayHandle);

        var labelErr = OpenVR.Overlay.CreateOverlay($"Label_{uniqueKey}", $"Label_{uniqueKey}", ref nameLabelOverlayHandle);
        if (labelErr != EVROverlayError.None) { Debug.LogError($"[FaceUIManager] Name Label overlay creation failed for {currentFaceName}: {labelErr}"); return; }
        OpenVR.Overlay.SetOverlayWidthInMeters(nameLabelOverlayHandle, nameLabelWidth);
        OpenVR.Overlay.SetOverlayAlpha(nameLabelOverlayHandle, 0f);
        OpenVR.Overlay.ShowOverlay(nameLabelOverlayHandle);

        if (Camera.main != null && nameLabelCanvas != null)
        {
            nameLabelCanvas.worldCamera = Camera.main;
        }
        else
        {
            Debug.LogError($"[FaceUIManager] Could not find Main Camera or NameLabelCanvas for '{currentFaceName}'! UI will not work correctly.");
        }
    }

    public void UpdateFaceData(Face faceData, Transform hmdTransform, HmdRotationState hmdState)
    {
        frameCounter++;

        if (hmdState == HmdRotationState.Moving)
        {
            return;
        }

        if (hmdState == HmdRotationState.Still)
        {
            // Reset anchor every X frames
            if (isAnchorSet && (frameCounter % update_each_X_frames == 0))
            {
                isAnchorSet = false;
            }

            if (!isAnchorSet)
            {
                float estimatedDistance = distanceFactor / faceData.h;
                estimatedDistance = Mathf.Clamp(estimatedDistance, 0.5f, 10.0f);
                float planeHalfWidth = Mathf.Tan(horizontalFov * Mathf.Deg2Rad * 0.5f) * estimatedDistance;
                float planeHalfHeight = Mathf.Tan(verticalFov * Mathf.Deg2Rad * 0.5f) * estimatedDistance;
                Vector3 localPos = new Vector3(
                    (faceData.x + (faceData.w * OFFSET)) * planeHalfWidth,
                    (faceData.y + (faceData.h * OFFSET)) * planeHalfHeight,
                    estimatedDistance
                );

                anchoredPosition = hmdTransform.TransformPoint(localPos);
                isAnchorSet = true;

                Color frameColor = Color.white;
                ColorUtility.TryParseHtmlString(faceData.color, out frameColor);

                if (frameTexture != null) Destroy(frameTexture);
                frameTexture = CreateFrameTexture(256, frameAspectRatio, frameColor, frameBorderWidth);

                var frameTexT = new Texture_t { handle = frameTexture.GetNativeTexturePtr(), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                OpenVR.Overlay.SetOverlayTexture(frameOverlayHandle, ref frameTexT);

                if (nameText != null) nameText.text = currentFaceName;
                if (backgroundPanel != null)
                {
                    Color correctedFrameColor = frameColor.gamma;
                    backgroundPanel.color = new Color(correctedFrameColor.r, correctedFrameColor.g, correctedFrameColor.b, 180f / 255f);
                }
            }
        }
        
        if (isAnchorSet)
        {
            if (nameLabelCamera != null)
            {
                nameLabelCamera.Render();
            }
            
            OpenVR.Overlay.SetOverlayAlpha(frameOverlayHandle, 1.0f);
            OpenVR.Overlay.SetOverlayAlpha(nameLabelOverlayHandle, 1.0f);

            Vector3 toCam = hmdTransform.position - anchoredPosition;
            if (toCam.sqrMagnitude > 1e-6f)
                anchoredRotation = Quaternion.LookRotation(-toCam, Vector3.up);

            var frameRigid = new SteamVR_Utils.RigidTransform(anchoredPosition, anchoredRotation);
            var frameMat = frameRigid.ToHmdMatrix34();
            OpenVR.Overlay.SetOverlayTransformAbsolute(frameOverlayHandle, ETrackingUniverseOrigin.TrackingUniverseStanding, ref frameMat);

            var bounds = new VRTextureBounds_t() { uMin = 0, uMax = 1, vMin = 1, vMax = 0 };
            OpenVR.Overlay.SetOverlayTextureBounds(nameLabelOverlayHandle, ref bounds);
            var labelTexT = new Texture_t { handle = nameLabelRenderTexture.GetNativeTexturePtr(), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
            OpenVR.Overlay.SetOverlayTexture(nameLabelOverlayHandle, ref labelTexT);

            float frameHalfHeight = (frameWorldWidth * frameAspectRatio) * 0.5f;
            Vector3 offsetVector = anchoredRotation * Vector3.up * (frameHalfHeight + nameLabelOffset);
            Vector3 labelPosition = anchoredPosition + offsetVector;

            var labelRigid = new SteamVR_Utils.RigidTransform(labelPosition, anchoredRotation);
            var labelMat = labelRigid.ToHmdMatrix34();
            OpenVR.Overlay.SetOverlayTransformAbsolute(nameLabelOverlayHandle, ETrackingUniverseOrigin.TrackingUniverseStanding, ref labelMat);
        }
    }

    private Texture2D CreateFrameTexture(int width, float aspectRatio, Color color, int borderWidth)
    {
        int height = Mathf.RoundToInt(width * aspectRatio);
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        Color[] pixels = new Color[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                pixels[y * width + x] = (x < borderWidth || x >= width - borderWidth || y < borderWidth || y >= height - borderWidth) ? color.linear : Color.clear;
            }
        }
        texture.SetPixels(pixels);
        texture.Apply();
        return texture;
    }
    

    public void Cleanup()
    {
        if (frameOverlayHandle == OpenVR.k_ulOverlayHandleInvalid && nameLabelOverlayHandle == OpenVR.k_ulOverlayHandleInvalid) return;

        Debug.Log($"[FaceUIManager] Cleaning up for '{currentFaceName}'. Destroying overlays.");
        if (frameOverlayHandle != OpenVR.k_ulOverlayHandleInvalid) OpenVR.Overlay.DestroyOverlay(frameOverlayHandle);
        if (nameLabelOverlayHandle != OpenVR.k_ulOverlayHandleInvalid) OpenVR.Overlay.DestroyOverlay(nameLabelOverlayHandle);

        frameOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;
        nameLabelOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;

        if (frameTexture != null) Destroy(frameTexture);
        if (nameLabelRenderTexture != null) Destroy(nameLabelRenderTexture);
    }

    void OnDestroy()
    {
        Cleanup();
    }
}
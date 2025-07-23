// FaceTrackingManager.cs

using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using Valve.VR;
using TMPro;
using UnityEngine.UI;

[System.Serializable]
public class Face
{
    public float x, y, w, h;
    public string name, color;
    public double timestamp;
}

[System.Serializable]
public class FaceList { public Face[] array; }

public enum HmdRotationState { Still, Moving }

public class FaceTrackingManager : MonoBehaviour
{
    [Header("Visual Alignment (Calibrated)")]
    public float horizontalFov = 100f;
    public float verticalFov = 75f;

    [Header("Sizing & Distance")]
    [Tooltip("A single factor to tune the distance. Calibrate this until 'Est. Dist' in logs matches reality.")]
    public float distanceFactor = 0.14f;
    [Tooltip("The real-world width of the frame in meters.")]
    public float frameWorldWidth = 0.2f;
    [Tooltip("The aspect ratio of the frame (Height / Width). >1 is portrait, <1 is landscape.")]
    public float frameAspectRatio = 1.3f;

    [Header("Frame Appearance")]
    public int frameBorderWidth = 10;
    public float OFFSET = 0.5f;

    [Header("HMD Movement Detection")]
    public float movementThreshold = 20f;
    public float stillThreshold = 0.5f;

    [Header("Multi-Face Management")]
    [Tooltip("The prefab that contains the UI for a single face.")]
    public GameObject faceUIPrefab;

    [Header("UI Elements")]
    public TextMeshProUGUI stateText;
    public Camera uiCamera;
    public RenderTexture uiRenderTexture;

    public int update_each_X_frames = 1;

    private ulong uiOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile Face[] latestFaces = new Face[0];

    private HmdRotationState currentHmdState = HmdRotationState.Still;
    private Quaternion lastHmdRotation;
    private float timeSpentStill = 0f;
    
    private Dictionary<string, FaceUIManager> activeFaces = new Dictionary<string, FaceUIManager>();
    private HashSet<string> facesSeenThisFrame = new HashSet<string>();
    private List<string> facesToRemove = new List<string>();
    private Dictionary<string,int> _nameCounters = new Dictionary<string,int>();
    private readonly HashSet<string> _facesSeenThisFrame = new HashSet<string>();
    private int faceCount = 0;


    void Start()
    {
                var initError = EVRInitError.None;
        // This is the CRUCIAL part. We are telling SteamVR that we are an OVERLAY application.
        OpenVR.Init(ref initError, EVRApplicationType.VRApplication_Overlay);

        if (initError != EVRInitError.None)
        {
            Debug.LogError($"[OpenVR] Initialization failed: {initError}");
            this.enabled = false;
            return;
        }

        if (faceUIPrefab == null)
        {
            Debug.LogError("[FaceTrackingManager] The 'Face UI Prefab' is not assigned in the Inspector! The script cannot proceed.", this);
            this.enabled = false;
            return;
        }

        SteamVR_Input.Initialize();
        udpClient = new UdpClient(5005);
        receiveThread = new Thread(ReceiveData) { IsBackground = true };
        receiveThread.Start();

        var err2 = OpenVR.Overlay.CreateOverlay("UIOverlay", "VR UI Overlay", ref uiOverlayHandle);
        if (err2 != EVROverlayError.None) { Debug.LogError($"[UI Overlay] Creation failed: {err2}"); }
        else
        {
            OpenVR.Overlay.SetOverlayAlpha(uiOverlayHandle, 1f);
            OpenVR.Overlay.SetOverlayWidthInMeters(uiOverlayHandle, 1.0f);
            OpenVR.Overlay.ShowOverlay(uiOverlayHandle);
        }

        if (Camera.main != null)
        {
            lastHmdRotation = Camera.main.transform.rotation;
        }
    }

    void ReceiveData()
    {
        var anyIP = new IPEndPoint(IPAddress.Any, 0);
        try
        {
            while (true)
            {
                byte[] data = udpClient.Receive(ref anyIP);
                var json = Encoding.UTF8.GetString(data);
                var wrapped = "{\"array\":" + json + "}";
                var faces = JsonUtility.FromJson<FaceList>(wrapped);
                latestFaces = (faces.array != null && faces.array.Length > 0) ? faces.array : new Face[0];
            }
        }
        catch (SocketException ex) { Debug.LogError("UDP Receive Error: " + ex.Message); }
    }

    void Update()
    {
        if (Camera.main == null) return;
        DetectHmdMovement();
        ManageFaceLifecycles();
        UpdateMainHUD();
    }

    void ManageFaceLifecycles()
    {
        if (currentHmdState != HmdRotationState.Still) return;

        _facesSeenThisFrame.Clear();
        _nameCounters.Clear();                 // reset counters every frame

        foreach (Face face in latestFaces)
        {
            if (string.IsNullOrEmpty(face.name)) continue;

            // BUILD A UNIQUE KEY 
            // this was integrated so the system can handel when the UDP package contains the same user 
            // (FR thinks the same face is detected multiple times)
            int idx = _nameCounters.TryGetValue(face.name, out int c) ? c : 0;
            string key = $"{face.name}_{idx}";
            _nameCounters[face.name] = idx + 1;

            _facesSeenThisFrame.Add(key);

            if (activeFaces.TryGetValue(key, out FaceUIManager mgr))
            {
                mgr.UpdateFaceData(face, Camera.main.transform, currentHmdState);
            }
            else
            {
                faceCount++;
                // Spawn a new UI
                GameObject go = Instantiate(faceUIPrefab);
                go.transform.position = new Vector3(faceCount * 2000.0f, 0, 0);
                mgr = go.GetComponent<FaceUIManager>();

                mgr.horizontalFov      = horizontalFov;
                mgr.verticalFov        = verticalFov;
                mgr.distanceFactor     = distanceFactor;
                mgr.OFFSET             = OFFSET;
                mgr.frameWorldWidth    = frameWorldWidth;
                mgr.frameAspectRatio   = frameAspectRatio;
                mgr.frameBorderWidth   = frameBorderWidth;
                mgr.nameLabelOffset    = 0.04f;
                mgr.nameLabelWidth     = 0.12f;

                mgr.Initialize(face, Camera.main.transform, key);

                activeFaces.Add(key, mgr);
            }
        }

        // ---------- DESTROY UI OBJECTS WE DIDN’T SEE THIS FRAME ----------
        facesToRemove.Clear();
        foreach (var kvp in activeFaces)
            if (!_facesSeenThisFrame.Contains(kvp.Key))
                facesToRemove.Add(kvp.Key);

        foreach (string k in facesToRemove)
        {
            Destroy(activeFaces[k].gameObject);
            activeFaces.Remove(k);
        }
    }
    
    void UpdateMainHUD()
    {
        if (stateText != null)
        {
            if (currentHmdState == HmdRotationState.Moving)
            {
                stateText.text = "State: " + currentHmdState.ToString();
            }
            else
            {
                stateText.text = "State:   " + currentHmdState.ToString();
            }
            
        }

        if (uiOverlayHandle != OpenVR.k_ulOverlayHandleInvalid && uiRenderTexture != null)
        {
            var bounds = new VRTextureBounds_t() { uMin = 0, uMax = 1, vMin = 1, vMax = 0 };
            OpenVR.Overlay.SetOverlayTextureBounds(uiOverlayHandle, ref bounds);
            var tex = new Texture_t { handle = uiRenderTexture.GetNativeTexturePtr(), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
            OpenVR.Overlay.SetOverlayTexture(uiOverlayHandle, ref tex);

            var mainCamera = Camera.main;
            if (mainCamera == null) return;

            var cam = Camera.main.transform;
            var offset = cam.forward * 0.4f;
            var position = cam.position + offset;
            var rigid = new SteamVR_Utils.RigidTransform(position, cam.rotation);
            var mat = rigid.ToHmdMatrix34();
            OpenVR.Overlay.SetOverlayTransformAbsolute(uiOverlayHandle, ETrackingUniverseOrigin.TrackingUniverseStanding, ref mat);
        }
    }
    
    private void DetectHmdMovement()
    {
        Quaternion currentHmdRotation = Camera.main.transform.rotation;
        float angleDifference = Quaternion.Angle(lastHmdRotation, currentHmdRotation);
        float angularSpeed = angleDifference / Time.deltaTime;

        if (angularSpeed > movementThreshold)
        {
            if (currentHmdState == HmdRotationState.Still)
            {
                Debug.Log("[Movement] State changed to MOVING.");
            }
            currentHmdState = HmdRotationState.Moving;
            timeSpentStill = 0f;
        }
        else
        {
            timeSpentStill += Time.deltaTime;
            if (currentHmdState == HmdRotationState.Moving && timeSpentStill >= stillThreshold)
            {
                Debug.Log("[Movement] State changed to STILL.");
                currentHmdState = HmdRotationState.Still;
            }
        }
        
        lastHmdRotation = currentHmdRotation;
    }

    void OnApplicationQuit()
    {
        if (uiOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
            OpenVR.Overlay.DestroyOverlay(uiOverlayHandle);

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }
        if (udpClient != null)
        {
            udpClient.Close();
        }
        
        OpenVR.Shutdown();
    }
}
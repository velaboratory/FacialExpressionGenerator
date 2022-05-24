using UnityEngine;
using System.Collections;


public class LeapShader : MonoBehaviour
{
    public Shader awesomeShader = null;
    private Material m_renderMaterial;
    Texture2D photo;
    public byte[] captureData;
    public bool doCapture = false;
    void Start()
    {
        if(transform.position.x == -0.02)
        {
            //dumb way to set the variation for the other 5 degree.
            //transform.position = new Vector3(-0.02f,Random.Range(-0.02f, 0.02f), Random.Range(-0.02f, 0.02f));
            //transform.rotation = Quaternion.Euler(Random.Range(-0.02f, 0.02f), Random.Range(-0.02f, 0.02f),
            //                                    Random.Range(-0.02f, 0.02f));
        }
        else
        {
            //transform.position = new Vector3(0.02f, Random.Range(-0.02f, 0.02f), Random.Range(-0.02f, 0.02f));
            //transform.rotation = Quaternion.Euler(Random.Range(-0.02f, 0.02f), Random.Range(-0.02f, 0.02f),
            //                                     Random.Range(-0.02f, 0.02f));
        }

        
        photo = new Texture2D(640, 480);
        if (awesomeShader == null)
        {
            Debug.LogError("no awesome shader.");
            m_renderMaterial = null;
            return;
        }
        m_renderMaterial = new Material(awesomeShader);
    }
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(source, destination, m_renderMaterial);

        if (doCapture)
        {
            RenderTexture save = RenderTexture.active;
            RenderTexture.active = destination;
            photo.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
            photo.Apply();
            RenderTexture.active = save;
            captureData = photo.EncodeToPNG();
            doCapture = false;
           
        }

    }
}
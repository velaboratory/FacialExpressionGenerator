using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LeapMotion : MonoBehaviour
{
    public LeapShader leapShaderLeft;
    public LeapShader leapShaderRight;
    public byte[] leftImage;
    public byte[] rightImage;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public IEnumerator captureImages()
	{
        
        leapShaderLeft.doCapture = true;
        leapShaderRight.doCapture = true;
        yield return new WaitUntil(() => !leapShaderRight.doCapture && !leapShaderLeft.doCapture);
        leftImage = leapShaderLeft.captureData;
        rightImage = leapShaderRight.captureData;
	}
}

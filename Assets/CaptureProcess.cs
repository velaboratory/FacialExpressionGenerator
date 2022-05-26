using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
public class CaptureProcess : MonoBehaviour
{
    public LeapMotion lm;
    public Avatar avatar;
    //public amountReset = 
    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(doCapture());
    }

    public string[] getBlendShapeNames(SkinnedMeshRenderer smr)
    {
        
        Mesh m = smr.sharedMesh;
        string[] arr;
        arr = new string[m.blendShapeCount];
        for (int i = 0; i < m.blendShapeCount; i++)
        {
            string s = m.GetBlendShapeName(i);
            arr[i] = s;
        }
        return arr;
    }

    public string getAvatarName(Avatar avatar)
    {
        string avatarName = avatar.gameObject.name;
        if (avatarName.Substring(0,1)=="M")
        {
            return "Male" + avatarName.Substring(avatarName.LastIndexOf("_") - 2, 2);
        }
        else
        {
            return "Female" + avatarName.Substring(avatarName.LastIndexOf("_") - 2, 2);
        }

        
    }

    public void RestartCoroutine()
    {
        StopCoroutine(doCapture());
        StartCoroutine(doCapture());
    }

    // Update is called once per frame
    IEnumerator doCapture()
	{
     
        string[] blendshapeNames = getBlendShapeNames(avatar.smr);
        string avatarName = getAvatarName(avatar);
        for (int i = 0; i < blendshapeNames.Length; i++)
		{
            //move the blendshape to 0, 25, 50, 75, 100, and then capture an image with the name
            for (int amount = 0; amount <= 100; amount += 25)
            {
                avatar.smr.SetBlendShapeWeight(i, amount);
                yield return null;
                yield return StartCoroutine(lm.captureImages());
                // print("test : "+ Application.persistentDataPath);
                //remove the Application.persistentDataPath +
                //File.WriteAllBytes("Output/rxp2left/leapLeft_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
                //File.WriteAllBytes("Output/rxp2right/leapRight_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);

                //File.WriteAllBytes( "Output/left/Female02leapLeft_b" + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
                //File.WriteAllBytes("Output/right/Female02leapRight_b" + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);
                avatar.smr.SetBlendShapeWeight(i, 0);
            }

            //also capture the 3d locations of the face markers
        }
        yield return null;
	}
}



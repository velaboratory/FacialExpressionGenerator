using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
public class CaptureProcess : MonoBehaviour
{
    public LeapMotion lm;
    public Avatar avatar;
    public string avatarName;
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
        //string avatarName = getAvatarName(avatar);
        for (int i = 0; i < blendshapeNames.Length; i++)
		{
            //move the blendshape to 0, 25, 50, 75, 100, and then capture an image with the name
            for (int amount = 0; amount <= 100; amount += 25)
            {
                avatar.smr.SetBlendShapeWeight(i, amount);
                yield return null;
                yield return StartCoroutine(lm.captureImages());
                
                //remove the Application.persistentDataPath +

                //HMD initial: Pos Vector3(0.575289965,0.990505934,-1.48304427), Rot Vector3.zero
                //RXn2Zp2. R means the HMD rotate, Xn2  in x negative 0.2, z positive 0.2. the rest hold as above
                //T means pos, Yp1 : Y positive 1.
                //File.WriteAllBytes("Output/TYp1RYn1/leapLeft_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
                //File.WriteAllBytes("Output/TYp1RYn1/leapRight_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);

                //File.WriteAllBytes( "Output/left/Female02leapLeft_b" + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
                //File.WriteAllBytes("Output/right/Female02leapRight_b" + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);
                avatar.smr.SetBlendShapeWeight(i, 0);
            }

            //also capture the 3d locations of the face markers
            //transform to cam space. leave
        }
        yield return null;
	}
}



using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Xml.Serialization;

public class CaptureProcess : MonoBehaviour
{
    public LeapMotion lm;
    public Avatar avatar;
    public string avatarName;
    string filePath = "Output";
    string file2ndaryPath = "/pic";
    bool writeIMG = true;
    public Camera LeapLeftCam;
    public Camera LeapRightCam;
    public Light Midlight;
    Vector3[] Landmarks;
    int lightIntensity;
    public Vector3 varPos;
    public Vector3 varRot;


    // Start is called before the first frame update
    void Start()
    {
        Landmarks = new Vector3[36];
        lightIntensity = Mathf.CeilToInt(Midlight.GetComponent<Light>().intensity * 10);
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

    public void disableLandmark(Avatar avatar)
    {
        for (int i = 0; i < avatar.transform.childCount; i++)
        {
            Transform child = avatar.transform.GetChild(i);
            if (child.name.Contains("Landmarks"))
            {
                //print(child.name);
                 child.gameObject.SetActive(false);
            }
        }
    }


    public Vector3[] getLandmarks(Avatar avatar,Camera cam)
    {
        Transform LandmarkerSets = null;
        for (int i = 0; i < avatar.transform.childCount; i++)
        {
            Transform child = avatar.transform.GetChild(i);
            if (child.name.Contains("Landmarks"))
            {
                //print(child.name);
                LandmarkerSets = child;
            }
        }
        Vector3[]  LandmarkVec = new Vector3[36];
        if(LandmarkerSets != null)
        {
            int indices = 0;
            foreach (Transform components in LandmarkerSets)
            {
                if (components.parent == LandmarkerSets)
                {
                    LandmarkVec[indices] = cam.WorldToScreenPoint(components.position);

                    //WorldToScreenPoint: The bottom-left of the screen is (0,0); the right-top is (pixelWidth,pixelHeight).

                    indices++;
                }
            }

        }


        return LandmarkVec;
    }

    public bool writeCSV(Vector3[] LandmarkVec, string filename)
    {
        StreamWriter xmlwriter = new StreamWriter(filename);
        foreach (Vector3 element in LandmarkVec)
        {
            xmlwriter.WriteLine(element.x+","+ (480- element.y) + "," + element.z);


        }


        xmlwriter.Close();
        return true;
    }

    public class landmarker
    {
        public float x;
        public float y;
        public float z;
    }
  
    public bool writeXML(Vector3[] LandmarkVec,string filename)
    {
        XmlSerializer xmler = new XmlSerializer(typeof(landmarker));
        StreamWriter xmlwriter = new StreamWriter(filename);
        landmarker markervector = new landmarker();


        foreach (Vector3 element in LandmarkVec)
        {
            markervector.x = element.x;
            markervector.y = element.y;
            markervector.z = element.z;
            xmler.Serialize(xmlwriter.BaseStream, markervector);

        }


        xmlwriter.Close();


        return true;
    }

    public void RestartCoroutine()
    {
        StopCoroutine("doCapture"); // needs to reference by string to specify it
        StartCoroutine("doCapture");
    }

    // Update is called once per frame
    public IEnumerator doCapture()
	{
      

        int[] indices = new int[] { 1, 2, 3, 4 ,5 ,6 ,7 ,8 ,9 ,10
            ,11 ,12 ,13 ,14 ,23 ,27 ,28 ,30 ,33 ,36 ,37 ,38 ,39 ,40
            ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,63 ,64 ,65 ,66 ,67 ,68
            ,69 ,70 ,71 ,72 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,89
            ,90 ,91 ,92 ,95 ,96 ,108 ,109}; // chosen for lower face proper deformation; see worklog

        string[] blendshapeNames = getBlendShapeNames(avatar.smr);


        if (writeIMG)
        {
            try
            {
                if (!Directory.Exists(filePath))
                {
                    Directory.CreateDirectory(filePath);
                }

            }
            catch (IOException ex)
            {
                Console.WriteLine(ex.Message);
            }
        }


        string nameLabel = "_r0_";
        int blendshapeBound = blendshapeNames.Length;
        disableLandmark(avatar);
        // in total 236 different pose of face deformation
        foreach (int element in indices)
        {
            if (element < blendshapeBound) //c# there is no for+if bundle? so assure within boundary  
            {
                for (int amount = 25; amount <= 100; amount += 25) // 0 is repeated across different blending
                {

                    avatar.smr.SetBlendShapeWeight(element, amount);
                    yield return null; //wait for the next frame and continue execution from this line
                    yield return StartCoroutine(lm.captureImages());

                    string filename= avatarName + blendshapeNames[element] + "_" + amount;

                    if (writeIMG)
                    {
                        File.WriteAllBytes(filePath + file2ndaryPath+ "/Light" + lightIntensity+ nameLabel
                            + "_leapLeft" + filename + ".png", lm.leftImage);
                        //writeXML(getLandmarks(avatar, LeapLeftCam), filename + "_leapLeft.xml");
                        File.WriteAllBytes(filePath + file2ndaryPath + "/Light" + lightIntensity + nameLabel
                            + "_leapRight" + filename + ".png", lm.rightImage);
                    }


                //also capture the 3d locations of the face markers
                //transform to cam space. leave. should be here before reset

                avatar.smr.SetBlendShapeWeight(element, 0); //reset !!! single variation. composite later
                }
               
                
            }
        }
       




        yield return null; // final closure thing for coroutine?

        /*
       for (int i = 0; i < blendshapeNames.Length; i++)
       {
           //move the blendshape to 0, 25, 50, 75, 100, and then capture an image with the name
           for (int amount = 0; amount <= 100; amount += 25)
           {
               avatar.smr.SetBlendShapeWeight(i, amount);
               yield return null; //wait for the next frame and continue execution from this line
               yield return StartCoroutine(lm.captureImages());

               //remove the Application.persistentDataPath +

               //HMD initial: Pos Vector3(0.575289965,0.990505934,-1.48304427), Rot Vector3.zero; in folder RXn2RZp2
               //if anything changed from initial! used the changed part as the name like following

               //RXn2Zp2. R means the HMD rotate, Xn2  in x negative 0.2, z positive 0.2. the rest hold as above
               //T means pos, Yp1 : Y positive 1.
               //File.WriteAllBytes("Output/I10/LI10RYp1TXp57leapLeft_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
               //File.WriteAllBytes("Output/I10/LI10RYp1TXp57leapRight_b" + avatarName + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);


               //File.WriteAllBytes( "Output/left/Female02leapLeft_b" + blendshapeNames[i] + "_" + amount + ".png", lm.leftImage);
               //File.WriteAllBytes("Output/right/Female02leapRight_b" + blendshapeNames[i] + "_" + amount + ".png", lm.rightImage);
               avatar.smr.SetBlendShapeWeight(i, 0);//reset !!!  just single variation. composite later
           }

           //also capture the 3d locations of the face markers
           //transform to cam space. leave
       }*/
    }
}



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
    public int ordinalAvatar;
    string filePath = "Output";
    string file2ndaryPath = "/AV8";
    string varLabel = "_v8_"; //see worklog for this label details of variation like hmd 6dof or lighting
    bool writeIMG = true;
    public Camera LeapLeftCam;
    public Camera LeapRightCam;
    public Light Midlight;
    Vector3[] Landmarks;
    int lightIntensity;
    public Vector3 varPos;
    public Vector3 varRot;
    public Transform lipsmark = null;

    int[] indices = new int[] { 1, 2, 3, 4 ,5 ,6 ,7 ,8 ,9 ,10
            ,11 ,12 ,13 ,14 ,23 ,27 ,28 ,30 ,33 ,36 ,37 ,38 ,39 ,40
            ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,63 ,64 ,65 ,66 ,67 ,68
            ,69 ,70 ,71 ,72 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,89
            ,90 ,91 ,92 ,95 ,96 ,108 ,109}; // chosen for lower face proper deformation; see worklog
    int[] avaindex = new int[] {16, 25, 32, 51, 55, 87, 89, 92, 95, 96}; //10 hold for test only
    private bool validfile = true;

    // Start is called before the first frame update
    void Start()
    {
        //Landmarks = new Vector3[36];
        //lightIntensity = Mathf.CeilToInt(Midlight.GetComponent<Light>().intensity * 10);
        //StartCoroutine(doCapture());
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
        /*
        Transform LandmarkerSets = null;
        for (int i = 0; i < avatar.transform.childCount; i++)
        {
            Transform child = avatar.transform.GetChild(i);
            if (child.name.Contains("Landmarks"))
            {
                //print(child.name);
                LandmarkerSets = child;
            }
        }*/
        //Vector3[]  LandmarkVec = new Vector3[36];
        Vector3[] LandmarkVec = new Vector3[4];
        if (lipsmark != null)
        {
            int indices = 0;
            foreach (Transform components in lipsmark)
            {
                if (components.parent == lipsmark)
                {
                    LandmarkVec[indices] = cam.WorldToScreenPoint(components.position);

                    //WorldToScreenPoint: The bottom-left of the screen is (0,0);
                    //   the right-top is (pixelWidth,pixelHeight). will be taken care of in writeCSV()

                    indices++;
                }
            }

        }


        return LandmarkVec;
    }

    public bool writeCSV(Vector3[] LandmarkVec, string filename)
    {
        StreamWriter CSVwriter = new StreamWriter(filename);
        foreach (Vector3 element in LandmarkVec)
        {
            CSVwriter.WriteLine(element.x + "," + (480 - element.y)); // + "," + element.z); // z for future
            //480 is the snapshot 

        }

        CSVwriter.Close();
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

        string[] blendshapeNames = getBlendShapeNames(avatar.smr);

        if  (Array.IndexOf(avaindex, ordinalAvatar) > -1)
        {
            validfile = false; //meaning this is the test data
        }
        else
        {
            validfile = true; //meaning this is training or valid data
        }

        int blendshapeBound = blendshapeNames.Length;
        //disableLandmark(avatar);
        // in total 236 different pose of face deformation
        foreach (int element in indices)
        {
            if (element < blendshapeBound) //c# there is no for+if bundle? so assure within boundary  
            {
                for (int amount = 15; amount <= 90; amount += 25) // 0 is repeated across different blending
                {

                    avatar.smr.SetBlendShapeWeight(element, amount);
                    yield return null; //wait for the next frame and continue execution from this line
                    //yield return StartCoroutine(lm.captureImages()); // this cause memory explosion?
                    //yield return lm.captureImages();

                    yield return lm.StartCoroutine("captureImages");                  
                    //yield return lm.startCaptureCoroutine();
                    string filename= ordinalAvatar + avatarName + blendshapeNames[element] + "_" + amount;

                    if (writeIMG )
                    {
                        if(validfile)
                        {
                            File.WriteAllBytes(filePath + file2ndaryPath + "/" + varLabel
                            + "_leapLeft" + filename + ".png", lm.leftImage);


                            writeCSV(getLandmarks(avatar, LeapLeftCam), filePath + file2ndaryPath + "/"
                                + varLabel + "_leapLeft" + filename + ".csv");

                            File.WriteAllBytes(filePath + file2ndaryPath + "/" + varLabel
                                + "_leapRight" + filename + ".png", lm.rightImage);

                            writeCSV(getLandmarks(avatar, LeapRightCam), filePath + file2ndaryPath + "/"
                                + varLabel + "_leapRight" + filename + ".csv");
                        }
                        else {
                            File.WriteAllBytes(filePath + "/test" + "/" + varLabel
                            + "_leapLeft" + filename + ".png", lm.leftImage);


                            writeCSV(getLandmarks(avatar, LeapLeftCam), filePath + "/test" + "/"
                                + varLabel + "_leapLeft" + filename + ".csv");

                            File.WriteAllBytes(filePath + "/test" + "/" + varLabel
                                + "_leapRight" + filename + ".png", lm.rightImage);

                            writeCSV(getLandmarks(avatar, LeapRightCam), filePath + "/test" + "/"
                                + varLabel + "_leapRight" + filename + ".csv");

                        }
                        
                    }


                //also capture the 3d locations of the face markers
                //transform to cam space. leave. should be here before reset

                avatar.smr.SetBlendShapeWeight(element, 0); //reset !!! single variation. composite later
                lm.StopCoroutine("captureImages");
                }
               
                
            }
        }
      
        yield return null; // wait for the next frame and continue execution from this line


    }
}



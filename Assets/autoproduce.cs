using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor; //for FindAssets()
using System.IO; //for directory()

public class autoproduce : MonoBehaviour
{
    //public GameObject[] prefabs;
    int indices = 0;
    
    //references
    public Transform avatarCliques;
    public CaptureProcess cp;
    public GameObject leftmouthmarker;
    public GameObject rightmouthmarker;
    private SkinnedMeshRenderer cacheSkin;
    public Transform HMDloction;
    Transform landmarks;
    Vector3 variation = new Vector3(0f, -0.005f, 0.005f);

    public struct avatarProp
    {
        //is there a better data container than struct? guess not, compiler has no data type inference 
        //use it for now to store the avatar name, initial pos

        public string name;
        public Vector3 bodyPos;
        public Vector3 leftMouthCornerPos;
        public Vector3 rightMouthCornerPos;

        public avatarProp(string names, Vector3 bPos, Vector3 lPos, Vector3 rPos) {

            // ....
            name = names;
            bodyPos = bPos;
            leftMouthCornerPos = lPos;
            rightMouthCornerPos = rPos;
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        //StartCoroutine("avatarLoadIteration"); //empty scene, load all avatars, drop it.
        StartCoroutine("avatarSimulation");
    
    }
    IEnumerator avatarSimulation()
    {
        //currently run sequentially; to run in paralell, all other scripts must be multiple instantiated.
        //float waitTime = 5; //was 5 seconds first round
         
        Transform[] avatarSets = avatarCliques.GetComponentsInChildren<Transform>(true);
        foreach (Transform avatar in avatarSets)
        {
            if(avatar.parent == avatarCliques && avatar.gameObject.activeSelf) // the first depth layer level
            {
                // 1. check it is active and adjust the pos
                // 2. check it does it have the avatar script: if not, attach one; set it to CP, set with mesh
                // 3. check it does have markers, if not, attach them, set with the mesh
                avatarProp mine = getAvatarProp(avatar.name);

                if (avatar.gameObject.activeSelf)
                {
                    // to set up this avatar property, deactivate to make sure nothing running to mess around
                    avatar.gameObject.SetActive(false);
                }

                //cp.disableLandmark(avatar.GetComponent<Avatar>());

                //print(avatar.name);
                //all avatars position are fixed and hmd is taylored, assigned hmd to the adjusted pos
                //Transform HMDtaylored = null;
                for (int i = 0; i < avatar.transform.childCount; i++)
                {
                    Transform child = avatar.transform.GetChild(i);
                    if (child.name.Contains("HMD"))
                    {
                        //print(child.name);
                        //HMDtaylored = child;
                        HMDloction.position = child.position + variation;
                    }
                }





                if (!avatar.GetComponent<Avatar>())
                {
                    avatar.gameObject.AddComponent<Avatar>();
                }
           
               

                cacheSkin = avatar.GetChild(1).GetComponent<SkinnedMeshRenderer>();
                avatar.GetComponent<Avatar>().smr = cacheSkin;
                //print(cacheSkin.gameObject.name);


                /*
                 * landmark so need for mouth marker
                bool LeftMouthHasMarker = false; //auto left marker always wrong. weird!
                bool RightMouthHasMarker = false;
                foreach (Transform components in avatar)
                {
                    if(components.parent == avatar)
                    {
                        if (components.name.Contains("LeftMouth"))
                        {
                            LeftMouthHasMarker = true;
                        }else if (components.name.Contains("RightMouth"))
                        {
                            RightMouthHasMarker = true;
                        }
                    }
                }
                if (!RightMouthHasMarker) 
                {
                    //
                 

                    GameObject rightmarker = Instantiate(rightmouthmarker);//,mine.rightMouthCornerPos,Quaternion.identity);
                    rightmarker.GetComponent<AttachToSkinnedMesh>().smr
                       = cacheSkin;
                    rightmarker.transform.parent = avatar;
                    rightmarker.transform.localPosition = mine.rightMouthCornerPos;
                    
                }
                if (!LeftMouthHasMarker)
                {
                    GameObject leftmarker = Instantiate(leftmouthmarker);//, mine.leftMouthCornerPos, Quaternion.identity);
                    leftmouthmarker.GetComponent<AttachToSkinnedMesh>().smr = cacheSkin;
                    //print("left one: " +cacheSkin.gameObject.name); //somehow editor side does not show properly
                    leftmarker.transform.parent = avatar;
                    //print("mine.leftMouthCornerPos " + mine.leftMouthCornerPos);
                    leftmarker.transform.localPosition = mine.leftMouthCornerPos;

                }*/
                // to animate 

                if (!avatar.gameObject.activeSelf)
                {
                    avatar.gameObject.SetActive(true);
                }

                cp.avatar = avatar.GetComponent<Avatar>();
                cp.avatarName = indices + mine.name;
                if (!cp.gameObject.activeSelf)
                {
                    cp.gameObject.SetActive(true);
                    
                }
                //cp.RestartCoroutine(); //wow deactivate/disable and enable/activate script won't restart the coroutine.
                yield return cp.doCapture();



                //yield return new WaitForSeconds(waitTime);

                avatar.gameObject.SetActive(false);
                indices += 1;

            }//check kids depth level
          
        }//foreach loop

        /*
        //the following cannot retrieve disabled/inactive 
        for (int i = 0; i < avatars.childCount; i++)
        {
            Transform child = avatars.GetChild(i);
            if (child.name.Contains("facial")){
                print(child.name);
            }

            yield return new WaitForSeconds(waitTime);
        }*/

    }

    IEnumerator avatarLoadIteration()
    {
        //dropped this could avoid the local folder guid difference problem
        //if they under resource folder, it could work, added to the scene during runtime.
        float waitTime = 10;
        string path = "Assets/Microsoft-Rocketbox/Assets/Avatars";
        if (Directory.Exists(path))
        {
            //fetch the local guid; test: 117 in total
            string[] guids = AssetDatabase.FindAssets("facial", new[] { path });

            //parse the guid to file path
            foreach (string guid in guids)
            {
                string avatarpath = AssetDatabase.GUIDToAssetPath(guid);
                avatarProp mine = getAvatarProp(avatarpath);

                Debug.Log(avatarpath + " : " + mine.name);

                //Object prefab = Resources.Load("../../"+avatarpath) ; // cannot use it,  must be in /Resources
                var myLoadedAssetBundle = AssetBundle.LoadFromFile(Path.Combine(Application.streamingAssetsPath,avatarpath));
                if (myLoadedAssetBundle == null)
                {
                    Debug.Log("Failed to load AssetBundle!");
                    yield break;
                    //return;
                }

                var prefab = myLoadedAssetBundle.LoadAsset<GameObject>("MyObject");

                GameObject t = (GameObject)Instantiate(prefab, new Vector3(0, 0, 0), Quaternion.identity);

                yield return new WaitForSeconds(waitTime);

                Destroy(t);



            }

            //naive to get the property


            //instantiate the obj
            // *** mark the mouth
            // *** set all the reference including the skin and capture/avatar
            // *** 


            //activate the everything and run the simulation



            //
            //
            //int datalen = guids.Length;
            // new GameObject[datalen];
        }





    }

    public avatarProp getAvatarProp(string avatarpath)
    {
        //a naive way to get a name this because adults may share the same pos to HMD
        if (avatarpath.Contains("Male") && !avatarpath.Contains("Child")) // avatarpath.Contains("Adult"))
        {
            avatarProp mine = new avatarProp("MaleAdult",
                new Vector3(0.772f, -0.07f, -0.09f), //Vector3(0.769999981,-0.0700000003,-0.0799999982)
                  new Vector3(-0.0244f, 1.614f, 0.122f),
                 new Vector3(0.0238f, 1.612f, 0.115f));
                
            
            return mine;
        }else if (avatarpath.Contains("Female") && !avatarpath.Contains("Child")) // avatarpath.Contains("Adult"))
        {
            avatarProp mine = new avatarProp("FemaleAdult",
                                new  Vector3(0.770f, 0, 0),
                                 new Vector3(-0.02182f, 1.5450f, 0.055f),
                                 new Vector3(0.02275f, 1.5485f, 0.05478f));
            return mine;
        }
        else if(avatarpath.Contains("Male"))
        {//kids
            avatarProp mine = new avatarProp("MaleKids",
                              new Vector3(0.77f, 0.27f, -0.06f),
                              new Vector3(-0.02018f, 1.27332f, 0.09285f),
                              new Vector3(0.02002f, 1.27332f, 0.0956f));
            return mine;
        }
        else
        {
            avatarProp mine = new avatarProp("FemaleKids",
                              new Vector3(0.77f, 0.275f, 0),
                              new Vector3(-0.022f, 1.272f, 0.03836f),
                              new Vector3(0.0182f, 1.272f, 0.0411f));
            return mine;
        }
    }//getAvatarProp




    // Update is called once per frame
    void Update()
    {
        
    }
}

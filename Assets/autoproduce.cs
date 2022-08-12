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
    Transform[] avatarSets;
    public CaptureProcess cp;
    public bool showLandmark=false;
    //public GameObject leftmouthmarker;
    //public GameObject rightmouthmarker;
    private SkinnedMeshRenderer cacheSkin;
    public Transform HMDloction;
    public Transform lipsmark = null;
    //Transform landmarks;
    Vector3 variation = new Vector3(0f, +0.002f, +0.0017f);
    Vector3 VariantRot = new Vector3(0.0f, 0.0f, 0.0f);
    avatarProp mine;

    public struct avatarProp
    {
        //is there a better data container than struct? guess not, compiler has no data type inference 
        //use it for now to store the avatar name, initial pos

        public string name;
        public Vector3 bodyPos;
        //public Vector3 leftMouthCornerPos;
        //public Vector3 rightMouthCornerPos;


        public avatarProp(string names, Vector3 bPos, Vector3 lPos, Vector3 rPos) {

            // ....
            name = names;
            bodyPos = bPos;
            //leftMouthCornerPos = lPos;
            //rightMouthCornerPos = rPos;
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        avatarSets = avatarCliques.GetComponentsInChildren<Transform>(true);
        //StartCoroutine("avatarLoadIteration"); //empty scene, load all avatars, drop it.
       
            StartCoroutine("avatarSimulation");
           // StopCoroutine("avatarSimulation");
       
        

    
    }
    IEnumerator avatarSimulation()
    {
    //IEnumerator: across frames also need to invoke the inner coroutine cp.doCapture();
    
       
        foreach (Transform avatar in avatarSets)
        {
            if(avatar.parent == avatarCliques) // the first depth layer level
            {
                // option for test:    && avatar.gameObject.activeSelf
                // 1. check it is active and adjust the pos
                // 2. check it does it have the avatar script: if not, attach one; set it to CP, set with mesh
                // 3. check it does have markers, if not, attach them, set with the mesh
                mine = getAvatarProp(avatar.name);

                if (avatar.gameObject.activeSelf)
                {
                    // to set up this avatar property, deactivate to make sure nothing running to mess around
                    avatar.gameObject.SetActive(false);
                }

                //cp.disableLandmark(avatar.GetComponent<Avatar>());


                //all avatars position are fixed and hmd is taylored, assigned hmd to the adjusted pos
                //Transform HMDtaylored = null;
                
                for (int i = 0; i < avatar.transform.childCount; i++)
                {
                    Transform child = avatar.transform.GetChild(i);
                    if (child.name.Contains("HMD"))
                    {
                        
                        //HMDtaylored = child;
                        HMDloction.position = child.position + variation;
                        HMDloction.transform.eulerAngles = VariantRot;
                    }else if (child.name.Contains("Lipsmark") && child.parent == avatar&&false)
                    {
                        //lipsmark = child;
                        cp.lipsmark = child;
                        foreach (Transform eachmark in child)
                        {
                            foreach (Transform vizSphere in eachmark)
                            {
                                vizSphere.gameObject.SetActive(false);
                                //deactivate the visual cue before running it.
                                //those only for aligning and testing
                            }
                        }



                    }
                }





                if (!avatar.GetComponent<Avatar>())
                {
                    avatar.gameObject.AddComponent<Avatar>();
                }
           
               
                
                cacheSkin = avatar.GetChild(1).GetComponent<SkinnedMeshRenderer>();
                if (cacheSkin!= avatar.GetComponent<Avatar>().smr)
                {
                    avatar.GetComponent<Avatar>().smr = cacheSkin;
                }
                   
                 

                // to animate 

                if (!avatar.gameObject.activeSelf)
                {
                    
                    avatar.gameObject.SetActive(true);
                }

                cp.avatar = avatar.GetComponent<Avatar>();
                cp.avatarName = mine.name;
                cp.ordinalAvatar = indices;
                if (!cp.gameObject.activeSelf)
                {
                    
                    cp.gameObject.SetActive(true);
                    
                }
                yield return cp.StartCoroutine("doCapture");
                //cp.RestartCoroutine(); //wow deactivate/disable and enable/activate script won't restart the coroutine.
                //yield return cp.doCapture();
                
                
                avatar.gameObject.SetActive(false);
                //Destroy(avatar.GetComponent<Avatar>()); //to avoid memory explosion? no
                //Destroy(avatar.gameObject);//to avoid memory explosion? no
                indices += 1;
                cp.StopCoroutine("doCapture");
                //System.GC.Collect();

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
        yield return null;
        
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

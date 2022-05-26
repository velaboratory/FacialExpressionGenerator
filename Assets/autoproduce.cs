using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor; //for FindAssets()
using System.IO; //for directory()

public class autoproduce : MonoBehaviour
{
    //public GameObject[] prefabs;
    //int indices = 0;
    
    //references
    public Transform avatarCliques;
    public CaptureProcess cp;
    public GameObject leftmouthmarker;
    public GameObject rightmouthmarker;

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
        float waitTime = 3;
         
        Transform[] avatarSets = avatarCliques.GetComponentsInChildren<Transform>(true);
        foreach (Transform avatar in avatarSets)
        {
            if(avatar.parent == avatarCliques) // the first depth layer level
            {
                // 1. check it is active and adjust the pos
                // 2. check it does it have the avatar script: if not, attach one; set it to CP, set with mesh
                // 3. check it does have markers, if not, attach them, set with the mesh
                avatarProp mine = getAvatarProp(avatar.name);
                if (!avatar.gameObject.activeSelf)
                {
                    avatar.gameObject.SetActive(true);                    
                }
                //print(avatar.name);
                //avatar.position = mine.bodyPos; //set the local pos
                avatar.transform.position = mine.bodyPos; //set the global pos


                if (!avatar.GetComponent<Avatar>())
                {
                    avatar.gameObject.AddComponent<Avatar>();
                }
           
                cp.avatar = avatar.GetComponent<Avatar>();


                avatar.GetComponent<Avatar>().smr = avatar.GetChild(1).GetComponent<SkinnedMeshRenderer>();

                if (avatar.childCount == 2)
                {
                    
                    GameObject leftmarker = Instantiate(leftmouthmarker, mine.leftMouthCornerPos,Quaternion.identity);
                    leftmouthmarker.GetComponent<AttachToSkinnedMesh>().smr
                        = avatar.GetChild(1).GetComponent<SkinnedMeshRenderer>();
                    leftmarker.transform.parent= avatar;
                    
                    GameObject rightmarker = Instantiate(rightmouthmarker,mine.rightMouthCornerPos,Quaternion.identity);
                    rightmarker.GetComponent<AttachToSkinnedMesh>().smr
                       = avatar.GetChild(1).GetComponent<SkinnedMeshRenderer>();
                    rightmarker.transform.parent = avatar;
                             
                }
                // to animate 
               yield return new WaitForSeconds(waitTime);
               avatar.gameObject.SetActive(false);
            }
          
        }
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
        float waitTime = 5;
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
                new Vector3(0.772f, -0.07f, -0.09f),
                  new Vector3(1, 1, 1),
                 new Vector3(1, 1, 1));
                
            
            return mine;
        }else if (avatarpath.Contains("Female") && !avatarpath.Contains("Child")) // avatarpath.Contains("Adult"))
        {
            avatarProp mine = new avatarProp("FemaleAdult",
                                new  Vector3(0.772f, 0, 0),
                                 new Vector3(1, 1, 1),
                                 new Vector3(1, 1, 1));
            return mine;
        }
        else if(avatarpath.Contains("Male"))
        {//kids
            avatarProp mine = new avatarProp("MaleKids",
                              new Vector3(1, 1, 1),
                              new Vector3(1, 1, 1),
                              new Vector3(1, 1, 1));
            return mine;
        }
        else
        {
            avatarProp mine = new avatarProp("FemaleKids",
                              new Vector3(1, 1, 1),
                              new Vector3(1, 1, 1),
                              new Vector3(1, 1, 1));
            return mine;
        }
    }//getAvatarProp




    // Update is called once per frame
    void Update()
    {
        
    }
}

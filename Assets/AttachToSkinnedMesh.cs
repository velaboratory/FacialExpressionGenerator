using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// transform that you think is really close to the target. The actual point will be found from this, and this transform will follow that point
/// </summary>
public class AttachToSkinnedMesh : MonoBehaviour
{
    int closest;
    Transform[] bones;
    float[] weights;
    Vector3[] offsets;
     
    [Tooltip("assigned to the skinned mesh renderer you want this point to follow")]
    public SkinnedMeshRenderer smr;

    [Tooltip("if checked, this would only use the bones, and would ignore blendshapes (only use if not using blendshapes!)")]
    public bool useBonesOnly;

    void Start()
    {
         
        Mesh m = new Mesh();
        smr.BakeMesh(m);
        //Debug.Log(m.vertices.Length);
        
        Vector3 p = smr.transform.InverseTransformPoint(this.transform.position);
        closest = 0;
        Vector3[] verts = m.vertices;
        float closestDistance = Vector3.Distance(verts[closest],p);
        for(int i = 1; i < m.vertices.Length; i++)
		{
            float d = Vector3.Distance(verts[i], p);
			if (d < closestDistance)
			{
                closestDistance = d;
                closest = i;
			}
        }

        BoneWeight boneWeights = smr.sharedMesh.boneWeights[closest];
        bones = new Transform[] {smr.bones[boneWeights.boneIndex0],
                                             smr.bones[boneWeights.boneIndex1],
                                             smr.bones[boneWeights.boneIndex2],
                                             smr.bones[boneWeights.boneIndex3]};
        weights = new float[] { boneWeights.weight0, boneWeights.weight1, boneWeights.weight2, boneWeights.weight3 };
        offsets = new Vector3[weights.Length];
        Vector3 originalPoint = smr.transform.TransformPoint(smr.sharedMesh.vertices[closest]);
        
        for (int i = 0; i < weights.Length; i++)
		{
            offsets[i] = bones[i].InverseTransformPoint(originalPoint);
		}
        Destroy(m);
    }

	Vector3 transformByBones()
	{
        Vector3 p = bones[0].TransformPoint(offsets[0]) * weights[0];

        for (int i = 1; i < weights.Length; i++)
		{
            p += bones[i].TransformPoint(offsets[i]) * weights[i];

        }
        return p;
	}
    private void OnDestroy()
    {
        
    }
    // Update is called once per frame
    void Update()
    {
        
       if (useBonesOnly)
       {
           Vector3 temp = transformByBones();
           this.transform.position = temp;
       }
       else
       {
           Mesh m = new Mesh();
           smr.BakeMesh(m);
           Vector3[] v = m.vertices;
           this.transform.position = smr.transform.TransformPoint(v[closest]);
           Destroy(m);
       }
        
       
    }
}

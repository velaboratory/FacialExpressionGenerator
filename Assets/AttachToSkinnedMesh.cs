using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(SkinnedMeshRenderer))]
public class AttachToSkinnedMesh : MonoBehaviour
{
    //a transform that you think is really close to the target.  The actual point will be found from this
    //this will also be what follows around the point
    public Transform closestPointAtStart;
    // Start is called before the first frame update
    SkinnedMeshRenderer smr;
    int closest;
    void Start()
    {
        smr = GetComponent<SkinnedMeshRenderer>();
        Mesh m = new Mesh();
        smr.BakeMesh(m);
        Debug.Log(m.vertices.Length);

        Vector3 p = transform.InverseTransformPoint(closestPointAtStart.transform.position);
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

        Debug.Log(closest + "," + closestDistance);
        
    }

 //   Vector3 transformByBones(int VertexIndex)
 //   {
 //       BoneWeight weights = smr.sharedMesh.boneWeights[VertexIndex];

 //       Transform[] b = new Transform[] {smr.bones[weights.boneIndex0],
 //                                            smr.bones[weights.boneIndex1],
 //                                            smr.bones[weights.boneIndex2],
 //                                            smr.bones[weights.boneIndex3]};
 //       float[] w = new float[] { weights.weight0, weights.weight1, weights.weight2, weights.weight3 };

 //       Vector3 p = smr.sharedMesh.vertices[VertexIndex];
 //       Vector3 newP = Vector3.zero;
 //       for(int i = 0; i < b.Length; i++)
	//	{
 //           newP += b[i].TransformPoint(p) * w[i];
	//	}
 //       return newP;
	//}
    // Update is called once per frame
    void Update()
    {
        Mesh m = new Mesh();
        smr.BakeMesh(m);
        Vector3[] v = m.vertices;
        closestPointAtStart.transform.position = this.transform.TransformPoint(v[closest]);
    }
}

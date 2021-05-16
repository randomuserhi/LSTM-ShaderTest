using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Main : MonoBehaviour
{
    LSTMManager.LSTMGroup LSTMGroup;
    ComputeShader LSTMShader;
    int Population = 500;

    public GameObject Agent;
    List<Agent> Agents = new List<Agent>();

    public GameObject TargetDisplay;
    public static Vector3 TargetPosition = new Vector3(10, 10);
    public int GenerationTimer = 500;
    public int MaxTimer = 500;

    public void Start()
    {
        Application.runInBackground = true;

        LSTMShader = LSTMManager.GenerateComputeShader();
        LSTMGroup = LSTMManager.CreateLSTMGroup(new int[] { 1, 3, 2, 1 }, Population);
        LSTMManager.AssignLSTMGroupToShader(LSTMGroup, LSTMShader);

        LSTMGroup.Initialize();

        for (int i = 0; i < LSTMGroup.WeightsBiases.Length; i++)
        {
            LSTMGroup.WeightsBiases[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        LSTMGroup.SetWeightBiasData();

        for (int i = 0; i < Population; i++)
        {
            GameObject NewAgent = Instantiate(Agent);
            NewAgent.transform.position = Vector2.zero;
            Agent A = NewAgent.GetComponent<Agent>();
            A.Network = i;
            A.Group = LSTMGroup;
            Agents.Add(A);
        }

        TargetPosition = new Vector3(UnityEngine.Random.Range(-40, 40), UnityEngine.Random.Range(-40, 40));
    }

    public float Speed = 1;
    public float MinSpeed = 1;
    public float Dist = 50;

    public void FixedUpdate()
    {
        if (TargetDisplay != null)
            TargetPosition = TargetDisplay.transform.position;

        LSTMManager.FeedForward(LSTMGroup.Inputs, LSTMGroup, LSTMShader);

        for (int i = 0; i < Agents.Count; i++)
        {
            Agents[i].SUpdate();
        }

        //Debug.Log(LSTMGroup.Outputs[0]);

        GenerationTimer -= 1;
        if (GenerationTimer < 0)
        {
            GenerationTimer = MaxTimer;

            Agents = Agents.OrderBy(A => A.Fitness).ToList();
            //Agents = Agents.OrderByDescending(A => A.Fitness).ToList();

            float Avg = Agents.Sum(A => A.Fitness) / Agents.Count;
            Debug.Log(Agents[0].Fitness + ", " + Agents[Agents.Count - 1].Fitness + " > " + Avg);

            Speed = Mathf.Min(Avg / 80f, 1);
            if (Speed > MinSpeed)
            {
                Speed = MinSpeed;
                MinSpeed += 0.1f;
            }
            else
            {
                MinSpeed = Speed;
            }

            for (int i = Agents.Count / 4; i < Agents.Count; i++)
            {
                int A = UnityEngine.Random.Range(0, Agents.Count / 4);
                Agents[i].transform.position = Agents[A].transform.position;
                Agents[i].transform.rotation = Agents[A].transform.rotation;
                Agents[i].Tail.transform.rotation = Agents[A].transform.rotation;
                Agents[i].Copy(Agents[A]);
                Agents[i].ResidualSpeed = Speed;
                Agents[i].Speed = 0;
                Agents[i].RB.velocity = Vector2.zero;

                LSTMGroup.Copy(Agents[A].Network, Agents[i].Network);
                LSTMGroup.Mutate(Agents[i].Network);
            }

            for (int i = 0; i < Agents.Count / 4; i++)
            {
                Agents[i].ResidualSpeed = Speed;
                Agents[i].Speed = 0;
                Agents[i].RB.velocity = Vector2.zero;
            }

            Quaternion R = new Quaternion();
            R.eulerAngles = new Vector3(0, 0, UnityEngine.Random.Range(0, 360));
            TargetDisplay.transform.position = R * new Vector3(0, Dist, 0) + new Vector3(Agents.Sum(A => A.transform.position.x) / Agents.Count, Agents.Sum(A => A.transform.position.y) / Agents.Count, 0);
            Vector3 position = TargetDisplay.transform.position;
            if (position.x > 80) position.x = -80;
            if (position.x < -80) position.x = 80;
            if (position.y > 80) position.y = -80;
            if (position.y < -80) position.y = 80;
            TargetDisplay.transform.position = position;
        }
    }

    public void OnApplicationQuit()
    {
        Debug.Log("Disposing");
        LSTMManager.DisposeGroup(LSTMGroup);
    }
}

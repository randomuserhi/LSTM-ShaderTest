using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Main : MonoBehaviour
{
    LSTMManager.LSTMGroup LSTMGroup;
    ComputeShader LSTM;
    float[] Inputs = new float[2 * 2];

    public void Start()
    {
        Application.runInBackground = true;

        LSTM = LSTMManager.GenerateComputeShader();
        LSTMGroup = LSTMManager.CreateLSTMGroup(new int[] { 2, 2 }, 2);
        LSTMManager.AssignLSTMGroupToShader(LSTMGroup, LSTM);
        LSTMGroup.Initialize();

        for (int i = 0; i < LSTMGroup.WeightsBiases.Length; i++)
        {
            LSTMGroup.WeightsBiases[i] = 1;
        }
        LSTMGroup.SetWeightBiasData();
        
        for (int i = 0; i < Inputs.Length; i++)
        {
            Inputs[i] = 1;
        }

        Debug.Log("Initialize Complete");
    }

    int Count = 0;
    public void FixedUpdate()
    {
        float[] Result = LSTMManager.FeedForward(Inputs, LSTMGroup, LSTM);
        Count++;
    }

    public void OnApplicationQuit()
    {
        Debug.Log("Disposing");
        LSTMManager.DisposeGroup(LSTMGroup);
    }
}

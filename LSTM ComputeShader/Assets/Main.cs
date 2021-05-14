using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class Main : MonoBehaviour
{
    LSTMManager.LSTMGroup LSTMGroup;

    public void Start()
    {
        ComputeShader LSTM = LSTMManager.GenerateComputeShader();
        LSTMGroup = LSTMManager.CreateLSTMGroup(new int[] { 10, 10, 10, 10, 10 }, 2000);
        LSTMManager.AssignLSTMGroupToShader(LSTMGroup, LSTM);
        LSTMGroup.Initialize();

        for (int i = 0; i < LSTMGroup.WeightsBiases.Length; i++)
        {
            LSTMGroup.WeightsBiases[i] = 1;
        }
        LSTMGroup.SetWeightBiasData();

        float[] Inputs = new float[2 * 2000];
        for (int i = 0; i < 2000; i++)
        {
            Inputs[i] = 1;
        }

        System.Diagnostics.Stopwatch S = new System.Diagnostics.Stopwatch();
        S.Start();

        for (int i = 0; i < 100; i++)
        {
            float[] Result = LSTMManager.FeedForward(Inputs, LSTMGroup, LSTM);
        }

        S.Stop();
        Debug.Log((S.ElapsedMilliseconds / 100) + "ms");
    }

    public void OnApplicationQuit()
    {
        Debug.Log("Disposing");
        LSTMManager.DisposeGroup(LSTMGroup);
    }
}

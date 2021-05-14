using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class LSTMEvol : _LSTM
{
    public LSTMEvol(int[] Structure) : base(Structure) { }
    public LSTMEvol(int[] Structure, LSTMEvol Copy) : base(Structure, Copy) { }

    public static LSTMEvol Merge(int[] Structure, LSTMEvol A, LSTMEvol B)
    {
        LSTMEvol Merged = new LSTMEvol(Structure, A);

        for (int i = 0; i < Merged.InputWeights.Length; i++)
        {
            MergeSingle(Merged.InputWeights[i], B.InputWeights[i]);
            MergeSingle(Merged.InputStateWeights[i], B.InputStateWeights[i]);
            MergeSingle(Merged.InputBias[i], B.InputBias[i]);
            MergeSingle(Merged.InputModulationWeights[i], B.InputModulationWeights[i]);
            MergeSingle(Merged.InputModulationStateWeights[i], B.InputModulationStateWeights[i]);
            MergeSingle(Merged.InputModulationBias[i], B.InputModulationBias[i]);
            MergeSingle(Merged.ForgetWeights[i], B.ForgetWeights[i]);
            MergeSingle(Merged.ForgetStateWeights[i], B.ForgetStateWeights[i]);
            MergeSingle(Merged.ForgetBias[i], B.ForgetBias[i]);
            MergeSingle(Merged.OutputWeights[i], B.OutputWeights[i]);
            MergeSingle(Merged.OutputStateWeights[i], B.OutputStateWeights[i]);
            MergeSingle(Merged.OutputBias[i], B.OutputBias[i]);
        }

        return Merged;
    }

    public static void MergeSingle(Matrix WeightsA, Matrix WeightsB)
    {
        WeightsA.GetData();
        WeightsB.GetData();
        for (int i = 0; i < WeightsA.Buffer.Length; i++)
        {
            float Chance = (float)R.NextDouble();
            if (Chance < 0.5f)
            {
                WeightsA.Buffer[i] = WeightsB.Buffer[i];
            }
        }
        WeightsA.SetData();
    }

    public void Mutate()
    {
        for (int i = 0; i < InputWeights.Length; i++)
        {
            MutateSingle(InputWeights[i]);
            MutateSingle(InputStateWeights[i]);
            MutateSingle(InputBias[i]);
            MutateSingle(InputModulationWeights[i]);
            MutateSingle(InputModulationStateWeights[i]);
            MutateSingle(InputModulationBias[i]);
            MutateSingle(ForgetWeights[i]);
            MutateSingle(ForgetStateWeights[i]);
            MutateSingle(ForgetBias[i]);
            MutateSingle(OutputWeights[i]);
            MutateSingle(OutputStateWeights[i]);
            MutateSingle(OutputBias[i]);
        }
    }

    private static void MutateSingle(Matrix Weight)
    {
        const float MutateChance = 0.001f;

        Weight.GetData();
        for (int i = 0; i < Weight.Buffer.Length; i++)
        {
            float Chance = (float)R.NextDouble();
            if (Chance < MutateChance)
            {
                float MutateOption = (float)R.NextDouble();
                if (MutateOption < 0.25f)
                    Weight.Buffer[i] *= -1f;
                else if (MutateOption < 0.5f)
                    Weight.Buffer[i] += 1f;
                else if (MutateOption < 0.75f)
                    Weight.Buffer[i] -= 1f;
                else
                    Weight.Buffer[i] = (float)R.NextDouble() * 2f - 1f;
            }

            float Variation = ((float)R.NextDouble() * 2f - 1f) * 0.01f;
            Weight.Buffer[i] += Variation;
        }
        Weight.SetData();
    }
}

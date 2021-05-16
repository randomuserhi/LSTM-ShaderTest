using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent : MonoBehaviour
{
    public int Network = 0;
    public LSTMManager.LSTMGroup Group;

    public GameObject Tail;
    public Rigidbody2D RB;

    float MaxRotationAcceleration = 1;
    public float AntiClockwiseStrength = 0.5f;

    // Start is called before the first frame update
    void Start()
    {
        RB = gameObject.GetComponent<Rigidbody2D>();   
    }

    public float ResidualSpeed = 1;
    public float Speed = 0;
    float PrevAngle = 0;
    public float Fitness;

    public void Reset()
    {
        Speed = 10;
        PrevAngle = 0;
        RB.velocity = Vector3.zero;
        RB.angularVelocity = 0;
        AntiClockwiseStrength = 0.5f;
        transform.position = Vector3.zero;
        transform.rotation = new Quaternion();
        Tail.transform.rotation = new Quaternion();
    }

    public void Copy(Agent A)
    {
        Speed = A.Speed;
        PrevAngle = A.PrevAngle;
        RB.velocity = A.RB.velocity;
        RB.angularVelocity = A.RB.angularVelocity;
        AntiClockwiseStrength = A.AntiClockwiseStrength;
    }

    bool Wait = true;

    // Update is called once per frame
    public void SUpdate()
    {
        if (Wait)
        {
            Wait = !Wait;
            return;
        }

        //Calculate Fitness
        Fitness = Vector3.Distance(transform.position, Main.TargetPosition);
        //Fitness = Vector3.Distance(transform.position, Vector3.zero);

        //Provide input
        int InputOffset = Group.GetInputOffset(Network);
        float A = Vector2.SignedAngle(transform.position - Main.TargetPosition, transform.rotation * Vector2.up) / 180;
        Group.Inputs[InputOffset] = A;
        //Group.Inputs[InputOffset + 1] = AntiClockwiseStrength * 2 - 1;

        //Read output
        int OutputOffset = Group.GetOutputOffset(Network);
        AntiClockwiseStrength = (Group.Outputs[OutputOffset] + 1f) / 2f;

        //Accelerate Tail
        Quaternion TailRot = Tail.transform.localRotation;
        PrevAngle = TailRot.eulerAngles.z;
        if (PrevAngle > 70) PrevAngle -= 360;

        float Angle = Group.Outputs[OutputOffset] * 70;

        TailRot.eulerAngles = new Vector3(0, 0, Angle);

        Tail.transform.localRotation = TailRot;

        //Rotate body
        RB.angularVelocity -= Angle * Mathf.Min(RB.velocity.magnitude / 2f, 1);

        //Accelerate body
        float ChangeInAngle = Math.Abs(Angle - PrevAngle);
        if (ChangeInAngle > 10)
        {
            Speed += ChangeInAngle * 0.1f;
        }
        Speed += ResidualSpeed;

        RB.velocity += (Vector2)(transform.rotation * new Vector2(0, Speed)) * Time.deltaTime;

        //Steer velocity vector towards body
        Angle = Vector2.SignedAngle(RB.velocity, transform.rotation * Vector2.up) * 0.3f;
        Angle *= Mathf.Deg2Rad;
        RB.velocity = new Vector2(Mathf.Cos(Angle) * RB.velocity.x - Mathf.Sin(Angle) * RB.velocity.y, Mathf.Sin(Angle) * RB.velocity.x + Mathf.Cos(Angle) * RB.velocity.y);
        
        RB.velocity *= 0.95f;
        Speed *= 0.95f;
        RB.angularVelocity *= 0.95f;
    }
}

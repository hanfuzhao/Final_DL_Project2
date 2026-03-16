import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

const SAFETYPE_API = "https://dl-project-2-second-version.onrender.com/api/predict";
const DASHBOARD_URL = "https://dlproject2-secondversion-production-029d.up.railway.app/#dashboard";

const SEVERITY_MAP: Record<string, string> = {
  racism: "high",
  threat: "high",
  hate_speech: "high",
  sexism: "high",
  cyberbullying: "high",
  toxicity: "medium",
  profanity: "medium",
  implicit_hate: "medium",
  sarcasm: "low",
  clean: "none",
};

interface Message {
  id: number;
  text: string;
  sender: string | null;
  app_source: string;
  direction: string;
  source_layer: string;
}

interface FlaggedResult {
  msgId: number;
  text: string;
  label: string;
  confidence: number;
  severity: string;
}

async function analyzeText(text: string): Promise<{
  label: string;
  confidence: number;
  is_flagged: boolean;
  severity: string;
}> {
  const resp = await fetch(SAFETYPE_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model: "distilbert" }),
  });

  if (!resp.ok) {
    throw new Error(`SafeType API error (${resp.status})`);
  }

  const data = await resp.json();
  const label = data.label || "clean";
  const confidence = data.confidence || 0;

  return {
    label,
    confidence,
    is_flagged: label !== "clean",
    severity: SEVERITY_MAP[label] || "low",
  };
}

async function sendAlertEmail(flaggedMessages: FlaggedResult[]) {
  const resendKey = Deno.env.get("RESEND_API_KEY");
  const parentEmail = Deno.env.get("PARENT_ALERT_EMAIL");

  if (!resendKey || !parentEmail) {
    console.warn("RESEND_API_KEY or PARENT_ALERT_EMAIL not set, skipping email");
    return;
  }

  const rows = flaggedMessages.map((m) => {
    const preview = m.text.length > 80 ? m.text.slice(0, 80) + "..." : m.text;
    const sevColor = m.severity === "high" ? "#dc2626" : m.severity === "medium" ? "#ea580c" : "#4f46e5";
    return `<tr>
      <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:${sevColor};font-weight:600;text-transform:capitalize">${m.label.replace(/_/g, " ")}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#6b7280;font-size:13px">${(m.confidence * 100).toFixed(1)}%</td>
      <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#374151;font-size:13px">${preview}</td>
    </tr>`;
  }).join("");

  const html = `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:600px;margin:0 auto">
      <div style="background:#4f46e5;color:#fff;padding:16px 24px;border-radius:8px 8px 0 0">
        <h2 style="margin:0;font-size:18px">SafeType Alert</h2>
        <p style="margin:4px 0 0;font-size:13px;opacity:0.85">${flaggedMessages.length} message(s) flagged for review</p>
      </div>
      <div style="border:1px solid #e5e7eb;border-top:none;border-radius:0 0 8px 8px;padding:20px 24px">
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr style="text-align:left">
              <th style="padding:8px 12px;border-bottom:2px solid #e5e7eb;font-size:12px;color:#6b7280;text-transform:uppercase">Category</th>
              <th style="padding:8px 12px;border-bottom:2px solid #e5e7eb;font-size:12px;color:#6b7280;text-transform:uppercase">Conf.</th>
              <th style="padding:8px 12px;border-bottom:2px solid #e5e7eb;font-size:12px;color:#6b7280;text-transform:uppercase">Message</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
        <div style="margin-top:20px;text-align:center">
          <a href="${DASHBOARD_URL}" style="display:inline-block;background:#4f46e5;color:#fff;padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:500;font-size:14px">View Dashboard</a>
        </div>
        <p style="margin-top:16px;font-size:12px;color:#9ca3af;text-align:center">This alert was sent by SafeType Parental Monitor</p>
      </div>
    </div>`;

  try {
    const resp = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${resendKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "SafeType Alerts <onboarding@resend.dev>",
        to: [parentEmail],
        subject: `SafeType: ${flaggedMessages.length} message(s) flagged`,
        html,
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      console.error(`Resend API error (${resp.status}): ${errText}`);
    } else {
      console.log(`Alert email sent to ${parentEmail}`);
    }
  } catch (err) {
    console.error("Failed to send alert email:", err);
  }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    let messageIds: number[] | null = null;
    try {
      const body = await req.json();
      if (body.message_ids && Array.isArray(body.message_ids)) {
        messageIds = body.message_ids;
      }
    } catch {
      // No body or invalid JSON
    }

    let query = supabase
      .from("messages")
      .select("id, text, sender, app_source, direction, source_layer")
      .is("is_flagged", null)
      .order("timestamp", { ascending: false })
      .limit(25);

    if (messageIds) {
      query = supabase
        .from("messages")
        .select("id, text, sender, app_source, direction, source_layer")
        .in("id", messageIds)
        .limit(25);
    }

    const { data: messages, error: fetchError } = await query;
    if (fetchError) {
      throw new Error(`Fetch error: ${fetchError.message}`);
    }

    if (!messages || messages.length === 0) {
      return new Response(
        JSON.stringify({ status: "ok", analyzed: 0, flagged: 0 }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    let flaggedCount = 0;
    let analyzedCount = 0;
    const flaggedResults: FlaggedResult[] = [];

    for (const msg of messages as Message[]) {
      try {
        const result = await analyzeText(msg.text);
        analyzedCount++;

        const flagReason = result.is_flagged
          ? `${result.label} (${(result.confidence * 100).toFixed(1)}% confidence)`
          : null;

        const { error: updateError } = await supabase
          .from("messages")
          .update({
            is_flagged: result.is_flagged,
            flag_reason: flagReason,
          })
          .eq("id", msg.id);

        if (updateError) {
          console.error(`Failed to update message ${msg.id}:`, updateError);
        }

        if (result.is_flagged) {
          flaggedCount++;
          flaggedResults.push({
            msgId: msg.id,
            text: msg.text,
            label: result.label,
            confidence: result.confidence,
            severity: result.severity,
          });
        }
      } catch (err) {
        console.error(`Failed to analyze message ${msg.id}:`, err);
      }
    }

    if (flaggedResults.length > 0) {
      await sendAlertEmail(flaggedResults);
    }

    return new Response(
      JSON.stringify({
        status: "ok",
        analyzed: analyzedCount,
        flagged: flaggedCount,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("Analysis error:", err);
    return new Response(
      JSON.stringify({ error: (err as Error).message }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

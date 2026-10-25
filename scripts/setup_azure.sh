#!/usr/bin/env bash
# One-time Azure + GitHub setup for the dengue early-warning MLOps pipeline.
# Run on your own computer, from the project folder, after `az login`:
#
#     bash scripts/setup_azure.sh
#
# Safe to re-run: anything that already exists is reused.
# Override any default like this:  LOC=centralindia WEBAPP=my-unique-name bash scripts/setup_azure.sh
set -euo pipefail
export PYTHONWARNINGS=ignore   # hide harmless SyntaxWarnings from the Azure CLI

GH_REPO="${GH_REPO:-Dinushi97/AML_Assignment_269133N_KumarasiriRMKD}"
RG="${RG:-rg-dengue-mlops}"
LOC="${LOC:-southeastasia}"          # Singapore; student subscriptions may require another region
WS="${WS:-mlw-dengue}"
WEBAPP="${WEBAPP:-dengue-ews-kumarasiri}"   # must be unique worldwide
PLAN="${PLAN:-plan-dengue}"
IDENTITY="${IDENTITY:-id-github-dengue}"

cd "$(dirname "$0")/.."
step() { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
trap 'printf "\n\033[1;31m✘ Failed at line %s.\033[0m\nIf the error says RequestDisallowedByAzure, re-run with an allowed region, e.g.  LOC=centralindia bash scripts/setup_azure.sh\nIf it is a quota error, edit azure/compute.yml (size: Standard_DS2_v2) and re-run.\n" "$LINENO"' ERR

step "Checking Azure login"
command -v az >/dev/null || { echo "Install the Azure CLI first:  brew install azure-cli"; exit 1; }
az account show >/dev/null 2>&1 || { echo "Run  az login  first (use your student account)."; exit 1; }
SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
echo "Signed in as $(az account show --query user.name -o tsv) | subscription: $(az account show --query name -o tsv)"
az extension add --name ml --yes --only-show-errors >/dev/null

step "Registering resource providers (first time can take a few minutes)"
PROVIDERS="MachineLearningServices Storage KeyVault Insights ContainerRegistry Web ManagedIdentity OperationalInsights"
for p in $PROVIDERS; do      # start all registrations; retry if the connection drops
  for _ in 1 2 3 4 5; do
    az provider register --namespace "Microsoft.$p" --only-show-errors >/dev/null 2>&1 && break
    echo "  network hiccup registering Microsoft.$p – retrying…"; sleep 10
  done
done
for _ in $(seq 1 60); do     # then wait (up to ~15 min) until all are Registered
  pending=""
  for p in $PROVIDERS; do
    state=$(az provider show --namespace "Microsoft.$p" --query registrationState -o tsv 2>/dev/null || echo "Unknown")
    [ "$state" = "Registered" ] || pending="$pending $p"
  done
  [ -z "$pending" ] && { echo "  all providers registered"; break; }
  echo "  still registering:$pending"; sleep 15
done

step "Resource group $RG ($LOC)"
az group create --name "$RG" --location "$LOC" --only-show-errors -o none

step "Azure Machine Learning workspace $WS"
if ! az ml workspace show --name "$WS" --resource-group "$RG" >/dev/null 2>&1; then
  az ml workspace create --name "$WS" --resource-group "$RG" --location "$LOC" --only-show-errors -o none
fi
az configure --defaults group="$RG" workspace="$WS"

step "Compute cluster cpu-cluster (scales to zero)"
az ml compute show --name cpu-cluster >/dev/null 2>&1 || az ml compute create --file azure/compute.yml --only-show-errors -o none

step "App Service plan (B1) and web app $WEBAPP"
az appservice plan show --name "$PLAN" --resource-group "$RG" >/dev/null 2>&1 \
  || az appservice plan create --name "$PLAN" --resource-group "$RG" --is-linux --sku B1 --location "$LOC" --only-show-errors -o none
if ! az webapp show --name "$WEBAPP" --resource-group "$RG" >/dev/null 2>&1; then
  az webapp create --name "$WEBAPP" --resource-group "$RG" --plan "$PLAN" --runtime "PYTHON:3.12" --only-show-errors -o none \
    || { echo "The name $WEBAPP may be taken. Re-run with WEBAPP=<another-unique-name>"; exit 1; }
fi
az webapp config set --name "$WEBAPP" --resource-group "$RG" --web-sockets-enabled true \
  --startup-file "python -m streamlit run app/app.py --server.port 8000 --server.address 0.0.0.0" --only-show-errors -o none
az webapp config appsettings set --name "$WEBAPP" --resource-group "$RG" \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000 --only-show-errors -o none

step "Managed identity for GitHub Actions (OIDC, no passwords)"
az identity show --name "$IDENTITY" --resource-group "$RG" >/dev/null 2>&1 \
  || az identity create --name "$IDENTITY" --resource-group "$RG" --location "$LOC" --only-show-errors -o none
CLIENT_ID=$(az identity show --name "$IDENTITY" --resource-group "$RG" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show --name "$IDENTITY" --resource-group "$RG" --query principalId -o tsv)
SCOPE="/subscriptions/$SUB/resourceGroups/$RG"
if [ -z "$(az role assignment list --assignee "$PRINCIPAL_ID" --scope "$SCOPE" --role Contributor --query '[0].id' -o tsv 2>/dev/null)" ]; then
  for _ in 1 2 3 4 5; do   # a brand-new identity can take a minute to appear
    az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
      --role Contributor --scope "$SCOPE" --only-show-errors -o none && break
    echo "  waiting for the identity to propagate…"; sleep 20
  done
fi
# GitHub may present the subject either by name or with immutable IDs (repo:Owner@ownerId/Repo@repoId:...); trust both
CREDS="github-main:repo:$GH_REPO:ref:refs/heads/main github-production:repo:$GH_REPO:environment:production"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  OWNER_ID=$(gh api "repos/$GH_REPO" --jq .owner.id 2>/dev/null || true)
  REPO_ID=$(gh api "repos/$GH_REPO" --jq .id 2>/dev/null || true)
  if [ -n "$OWNER_ID" ] && [ -n "$REPO_ID" ]; then
    ID_REPO="${GH_REPO%%/*}@$OWNER_ID/${GH_REPO#*/}@$REPO_ID"
    CREDS="$CREDS github-main-ids:repo:$ID_REPO:ref:refs/heads/main github-production-ids:repo:$ID_REPO:environment:production"
  fi
fi
for cred in $CREDS; do
  NAME=${cred%%:*}; SUBJECT=${cred#*:}
  az identity federated-credential show --name "$NAME" --identity-name "$IDENTITY" --resource-group "$RG" >/dev/null 2>&1 \
    || az identity federated-credential create --name "$NAME" --identity-name "$IDENTITY" --resource-group "$RG" \
         --issuer https://token.actions.githubusercontent.com --subject "$SUBJECT" \
         --audiences api://AzureADTokenExchange --only-show-errors -o none
done

step "GitHub secrets and variables for $GH_REPO"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh secret set AZURE_CLIENT_ID       --repo "$GH_REPO" --body "$CLIENT_ID"
  gh secret set AZURE_TENANT_ID       --repo "$GH_REPO" --body "$TENANT"
  gh secret set AZURE_SUBSCRIPTION_ID --repo "$GH_REPO" --body "$SUB"
  gh variable set AZURE_RESOURCE_GROUP --repo "$GH_REPO" --body "$RG"
  gh variable set AZURE_ML_WORKSPACE   --repo "$GH_REPO" --body "$WS"
  gh variable set AZURE_WEBAPP_NAME    --repo "$GH_REPO" --body "$WEBAPP"
  echo "Done: 3 secrets and 3 variables set on GitHub."
else
  cat <<EOF
GitHub CLI not logged in, so add these by hand:
github.com/$GH_REPO → Settings → Secrets and variables → Actions

  Secrets:    AZURE_CLIENT_ID        = $CLIENT_ID
              AZURE_TENANT_ID        = $TENANT
              AZURE_SUBSCRIPTION_ID  = $SUB
  Variables:  AZURE_RESOURCE_GROUP   = $RG
              AZURE_ML_WORKSPACE     = $WS
              AZURE_WEBAPP_NAME      = $WEBAPP
EOF
fi

trap - ERR
step "All set ✔"
echo "Now start the pipeline: GitHub → Actions → 'Azure MLOps – test, train, register, deploy' → Run workflow"
echo "Azure ML studio: https://ml.azure.com   ·   App (after the first deploy): https://$WEBAPP.azurewebsites.net"

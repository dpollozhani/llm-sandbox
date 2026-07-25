// Minimal illustrative IaC for hosting the assistant: a Container App running
// the Docker image from deploy/docker, backed by an Azure OpenAI account.
// Not a complete production template (no networking, monitoring, or Key
// Vault wiring) - a starting point to adapt to your subscription/landing zone.
@description('Name prefix for all resources')
param namePrefix string = 'data-analyst'

@description('Azure region')
param location string = resourceGroup().location

@description('Container image, e.g. myregistry.azurecr.io/data-analyst-assistant:latest')
param containerImage string

resource openAi 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: '${namePrefix}-openai'
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: '${namePrefix}-openai'
  }
}

resource openAiDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: openAi
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
    }
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env'
  location: location
  properties: {}
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-app'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
    }
    template: {
      containers: [
        {
          name: 'app'
          image: containerImage
          env: [
            { name: 'LLM_PROVIDER', value: 'azure_openai' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAi.properties.endpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: openAiDeployment.name }
          ]
        }
      ]
    }
  }
}

output appUrl string = containerApp.properties.configuration.ingress.fqdn

# Terraform equivalent of infrastructure/bicep/main.bicep - same scope and
# same caveats: illustrative starting point, not a complete production
# landing zone (no networking, monitoring, or secret management wired up).

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "name_prefix" {
  type    = string
  default = "data-analyst"
}

variable "location" {
  type    = string
  default = "westeurope"
}

variable "container_image" {
  type        = string
  description = "e.g. myregistry.azurecr.io/data-analyst-assistant:latest"
}

resource "azurerm_resource_group" "this" {
  name     = "${var.name_prefix}-rg"
  location = var.location
}

resource "azurerm_cognitive_account" "openai" {
  name                = "${var.name_prefix}-openai"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  kind                = "OpenAI"
  sku_name            = "S0"
}

resource "azurerm_cognitive_deployment" "gpt4o" {
  name                 = "gpt-4o"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format = "OpenAI"
    name   = "gpt-4o"
  }
  sku {
    name     = "Standard"
    capacity = 10
  }
}

resource "azurerm_container_app_environment" "this" {
  name                = "${var.name_prefix}-env"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
}

resource "azurerm_container_app" "this" {
  name                         = "${var.name_prefix}-app"
  resource_group_name          = azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  revision_mode                = "Single"

  template {
    container {
      name   = "app"
      image  = var.container_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "LLM_PROVIDER"
        value = "azure_openai"
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = azurerm_cognitive_account.openai.endpoint
      }
      env {
        name  = "AZURE_OPENAI_DEPLOYMENT"
        value = azurerm_cognitive_deployment.gpt4o.name
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

output "app_url" {
  value = azurerm_container_app.this.ingress[0].fqdn
}

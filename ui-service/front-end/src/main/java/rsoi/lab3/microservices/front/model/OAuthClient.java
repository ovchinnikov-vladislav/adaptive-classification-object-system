package rsoi.lab3.microservices.front.model;

import javax.validation.constraints.NotNull;
import java.io.Serializable;
import java.util.Date;
import java.util.UUID;

public class OAuthClient implements Serializable {
    private static final long serialVersionUID = 43127469381264723L;

    private UUID clientId;
    private String clientSecret;
    private String scope;
    @NotNull
    private String redirectUri;
    @NotNull
    private String applicationName;
    @NotNull
    private boolean grantPassword;
    private Date createdAt;
    private Date updatedAt;

    public UUID getClientId() {
        return clientId;
    }

    public void setClientId(UUID clientId) {
        this.clientId = clientId;
    }

    public String getClientSecret() {
        return clientSecret;
    }

    public void setClientSecret(String clientSecret) {
        this.clientSecret = clientSecret;
    }

    public String getScope() {
        return scope;
    }

    public void setScope(String scope) {
        this.scope = scope;
    }

    public String getRedirectUri() {
        return redirectUri;
    }

    public void setRedirectUri(String redirectUri) {
        this.redirectUri = redirectUri;
    }

    public String getApplicationName() {
        return applicationName;
    }

    public void setApplicationName(String applicationName) {
        this.applicationName = applicationName;
    }

    public boolean isGrantPassword() {
        return grantPassword;
    }

    public void setGrantPassword(boolean grantPassword) {
        this.grantPassword = grantPassword;
    }

    public Date getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Date createdAt) {
        this.createdAt = createdAt;
    }

    public Date getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Date updatedAt) {
        this.updatedAt = updatedAt;
    }
}

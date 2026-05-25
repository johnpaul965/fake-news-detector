import { FlaskConical, Filter, Tag } from "lucide-react";
import { useGetFeatureMeta } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { useState } from "react";

export default function Features() {
  const { data: features, isLoading } = useGetFeatureMeta();
  const [searchTerm, setSearchTerm] = useState("");

  const filteredFeatures = features?.filter(f => 
    f.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    f.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    f.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const linguisticCount = features?.filter(f => f.category === "linguistic").length || 0;
  const structuralCount = features?.filter(f => f.category === "structural").length || 0;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-2">
          <FlaskConical className="h-7 w-7 text-primary" />
          Feature Reference
        </h1>
        <p className="text-muted-foreground text-lg">
          Documentation of the {features?.length || 0} features extracted by the model during classification.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
        <div className="relative w-full sm:max-w-md">
          <Filter className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search features..." 
            className="pl-9 bg-card/50"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="flex gap-3 text-sm">
          <Badge variant="outline" className="bg-chart-1/10 text-chart-1 border-chart-1/20 px-3 py-1">
            {linguisticCount} Linguistic
          </Badge>
          <Badge variant="outline" className="bg-chart-2/10 text-chart-2 border-chart-2/20 px-3 py-1">
            {structuralCount} Structural
          </Badge>
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Skeleton key={i} className="h-32 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredFeatures?.map(feature => (
            <Card key={feature.name} className="shadow-sm bg-card/50 hover:border-primary/30 transition-colors">
              <CardHeader className="p-4 pb-2">
                <div className="flex justify-between items-start mb-2">
                  <Badge 
                    variant="outline" 
                    className={feature.category === "linguistic" 
                      ? "bg-chart-1/10 text-chart-1 border-chart-1/20" 
                      : "bg-chart-2/10 text-chart-2 border-chart-2/20"
                    }
                  >
                    {feature.category}
                  </Badge>
                  <Tag className="h-3 w-3 text-muted-foreground opacity-50" />
                </div>
                <CardTitle className="text-base leading-tight">{feature.label}</CardTitle>
                <CardDescription className="font-mono text-xs text-muted-foreground/80 mt-1">
                  {feature.name}
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-2 text-sm text-muted-foreground leading-relaxed">
                {feature.description}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
      
      {!isLoading && filteredFeatures?.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No features found matching "{searchTerm}"
        </div>
      )}
    </div>
  );
}